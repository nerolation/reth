#[cfg(all(feature = "test-utils", feature = "metrics"))]
pub mod timing_tests {
    use crate::{
        ConfigureEvm, Database, metrics::evm_metrics,
        execute::{BlockBuilder, BlockExecutionError},
    };
    use alloy_consensus::{Block, BlockHeader, Header, Receipt, TransactionSigned, TransactionSignedEcRecovered};
    use alloy_evm::block::BlockExecutorFactory;
    use alloy_primitives::{Address, B256, Bytes, U256};
    use metrics_util::debugging::{DebuggingRecorder, Snapshotter};
    use reth_ethereum_primitives::EthPrimitives;
    use reth_primitives_traits::{NodePrimitives, RecoveredBlock, SealedHeader};
    use revm::{
        context::TxEnv,
        database::{CacheDB, EmptyDB, State},
    };
    use std::{sync::Arc, time::Duration};

    /// Setup a debugging recorder for metrics tests
    pub fn setup_metrics_recorder() -> Snapshotter {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        recorder.install().unwrap();
        snapshotter
    }

    /// Create a mock transaction for testing
    pub fn create_test_transaction() -> TransactionSignedEcRecovered {
        // Create a simple transaction
        let tx = TransactionSigned {
            hash: B256::random(),
            signature: Default::default(),
            transaction: alloy_consensus::Transaction::Eip1559(alloy_consensus::Eip1559Transaction {
                chain_id: 1,
                nonce: 0,
                gas_limit: 21000,
                max_fee_per_gas: U256::from(30_000_000_000u64),
                max_priority_fee_per_gas: U256::from(1_000_000_000u64),
                to: alloy_primitives::TxKind::Call(Address::random()),
                value: U256::from(1_000_000_000_000_000_000u64), // 1 ETH
                access_list: Default::default(),
                input: Bytes::default(),
            }),
        };

        // Convert to recovered transaction
        TransactionSignedEcRecovered {
            hash: tx.hash,
            signature: tx.signature,
            transaction: tx.transaction,
            signer: Address::random(),
        }
    }

    /// Execute a transaction and verify timing metrics are recorded.
    ///
    /// This is a helper function that can be used by other tests to verify
    /// that transaction execution timing metrics are working correctly in
    /// different contexts.
    pub fn execute_transaction_and_verify_metrics<C>(
        config_evm: C,
        parent_header: SealedHeader<<EthPrimitives as NodePrimitives>::BlockHeader>,
    ) -> Result<(), BlockExecutionError>
    where
        C: ConfigureEvm<Primitives = EthPrimitives>,
    {
        // Setup metrics recorder
        let snapshotter = setup_metrics_recorder();
        
        // Create a transaction
        let tx = create_test_transaction();
        
        // Create a recovery transaction struct for the builder
        let tx_recovered = reth_primitives_traits::Recovered::new(tx, tx.signer);
        
        // Create a database and necessary state
        let db = CacheDB::<EmptyDB>::default();
        let mut state = State::builder().with_database(db).with_bundle_update().build();
        
        // Create a block builder
        let evm_env = config_evm.evm_env(&parent_header);
        let attributes = C::NextBlockEnvCtx::default();
        let ctx = config_evm.context_for_next_block(&parent_header, attributes);
        let evm = config_evm.evm_with_env(&mut state, evm_env);
        
        // Create a block builder
        let mut builder = config_evm.create_block_builder(evm, &parent_header, ctx);
        
        // Apply necessary changes before execution
        builder.apply_pre_execution_changes()?;
        
        // Execute the transaction and capture result
        let gas_used = builder.execute_transaction(tx_recovered)?;
        
        // Check metrics were recorded
        let sample_count_before = evm_metrics().transaction_execution_time.get_sample_count();
        assert!(sample_count_before > 0, "Transaction execution time metric should be recorded");
        
        // Get metrics snapshot and verify
        let snapshot = snapshotter.snapshot().into_vec();
        let mut found_transaction_timing = false;
        
        for metric in snapshot {
            let metric_name = metric.0.key().name();
            if metric_name == "evm.transaction_execution_time" {
                found_transaction_timing = true;
                break;
            }
        }
        
        assert!(found_transaction_timing, "Transaction execution time metric should be found in snapshot");
        
        // Execute another transaction and verify metrics are updated
        let tx2 = create_test_transaction();
        let tx2_recovered = reth_primitives_traits::Recovered::new(tx2, tx2.signer);
        let _ = builder.execute_transaction(tx2_recovered)?;
        
        let sample_count_after = evm_metrics().transaction_execution_time.get_sample_count();
        assert!(
            sample_count_after > sample_count_before, 
            "Transaction execution time metric should be updated after second transaction"
        );
        
        Ok(())
    }

    /// Test integrating transaction timing with a real BlockExecutor implementation
    #[test]
    #[cfg(test)]
    fn test_transaction_execution_timing_integrated() {
        // This test would use a real BlockExecutorFactory implementation
        // and would be part of a higher-level integration test
        // For now, we're providing a framework that can be used
        // in actual integration tests
    }
} 