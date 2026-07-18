//! Helpers for the `eth_getBlockAccessList` and `debug_getRawBlockAccessList` RPC methods.
use alloy_consensus::BlockHeader;
use alloy_eip7928::{bal::DecodedBal, BlockAccessList};
use alloy_primitives::Bytes;
use alloy_rpc_types_eth::BlockId;
use reth_chainspec::{ChainSpecProvider, EthereumHardforks};
use reth_errors::RethError;
use reth_evm::{block::BlockExecutor, ConfigureEvm, Evm};
use reth_revm::{database::StateProviderDatabase, State};
use reth_rpc_eth_types::{
    cache::db::StateProviderTraitObjWrapper, error::FromEthApiError, EthApiError,
};
use reth_storage_api::StateProviderFactory;

use crate::{
    helpers::{Call, LoadBlock, Trace},
    RpcNodeCore, RpcNodeCoreExt,
};

/// Helper trait for the `eth_getBlockAccessList` and `debug_getRawBlockAccessList` RPC methods.
pub trait GetBlockAccessList: Trace + Call + LoadBlock + RpcNodeCoreExt {
    /// Retrieves the block access list for the given block.
    ///
    /// Returns `None` if the block does not exist. Requesting a known pre-Amsterdam block is an
    /// error because block access lists are only defined from the Amsterdam hardfork onwards.
    fn get_block_access_list(
        &self,
        block_id: BlockId,
    ) -> impl Future<Output = Result<Option<BlockAccessList>, Self::Error>> + Send {
        async move {
            let Some(block) = self.recovered_block(block_id).await? else { return Ok(None) };

            // Gate before touching cache or re-executing: pre-Amsterdam blocks have no block
            // access list and must not be re-executed to synthesize one.
            if !self.provider().chain_spec().is_amsterdam_active_at_timestamp(block.timestamp()) {
                return Err(EthApiError::BlockAccessListNotAvailablePreAmsterdam.into())
            }

            if let Some(cached_bal) =
                self.cache().get_bal(block.hash()).await.map_err(Self::Error::from_eth_err)?
            {
                let (bal, _) = DecodedBal::from_rlp_bytes(cached_bal.as_raw().clone())
                    .map_err(RethError::other)
                    .map_err(Self::Error::from_eth_err)?
                    .split();
                return Ok(Some(Vec::from(bal)))
            }

            self.spawn_blocking_io(move |eth_api| {
                let state = eth_api
                    .provider()
                    .state_by_block_id(block.parent_hash().into())
                    .map_err(Self::Error::from_eth_err)?;

                let mut db = State::builder()
                    .with_database(StateProviderDatabase::new(StateProviderTraitObjWrapper(state)))
                    .with_bal_builder()
                    .build();

                let block_txs = block.transactions_recovered();
                let mut executor = RpcNodeCore::evm_config(&eth_api)
                    .executor_for_block(&mut db, block.sealed_block())
                    .map_err(RethError::other)
                    .map_err(Self::Error::from_eth_err)?;

                executor.apply_pre_execution_changes().map_err(Self::Error::from_eth_err)?;
                executor.evm_mut().db_mut().bump_bal_index();

                // replay all transactions prior to the targeted transaction
                for block_tx in block_txs {
                    executor.execute_transaction(block_tx).map_err(Self::Error::from_eth_err)?;
                    executor.evm_mut().db_mut().bump_bal_index();
                }

                executor
                    .apply_post_execution_changes()
                    .map_err(|err| EthApiError::Internal(err.into()))?;

                let bal = db.take_built_alloy_bal();
                Ok(Some(bal.unwrap_or_default()))
            })
            .await
        }
    }

    /// Retrieves the raw RLP-encoded block access list for a block.
    ///
    /// Unlike [`Self::get_block_access_list`] an unknown block is an error, and an empty block
    /// access list is encoded as `0xc0`.
    fn get_raw_block_access_list(
        &self,
        block_id: BlockId,
    ) -> impl Future<Output = Result<Bytes, Self::Error>> + Send {
        async move {
            let block = self
                .recovered_block(block_id)
                .await?
                .ok_or_else(|| EthApiError::HeaderNotFound(block_id))?;

            if let Some(cached_bal) =
                self.cache().get_bal(block.hash()).await.map_err(Self::Error::from_eth_err)?
            {
                return Ok(cached_bal.as_raw().clone())
            }

            let bal = self.get_block_access_list(block_id).await?.unwrap_or_default();
            Ok(alloy_rlp::encode(bal).into())
        }
    }
}
