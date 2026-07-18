//! Spec-shaped response types for the `eth_getBlockAccessList` RPC method.
//!
//! These mirror the EIP-7928 block access list schema of the
//! [`execution-apis`](https://github.com/ethereum/execution-apis) specification: access indices are
//! hex quantities and storage keys/values are padded 32-byte words, unlike the consensus encoding
//! exposed by [`alloy_eip7928`].

use alloy_primitives::{Address, Bytes, B256, U256, U64};
use serde::{Deserialize, Serialize};

/// Accesses and post-block state changes of a single account within a block access list.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AccountAccess {
    /// Address of the accessed account.
    pub address: Address,
    /// Storage writes, grouped per slot.
    pub storage_changes: Vec<SlotChanges>,
    /// Storage keys that were read without being written to.
    pub storage_reads: Vec<B256>,
    /// Balance changes of the account.
    pub balance_changes: Vec<BalanceChange>,
    /// Nonce changes of the account.
    pub nonce_changes: Vec<NonceChange>,
    /// Code changes of the account.
    pub code_changes: Vec<CodeChange>,
}

impl From<alloy_eip7928::AccountChanges> for AccountAccess {
    fn from(changes: alloy_eip7928::AccountChanges) -> Self {
        Self {
            address: changes.address,
            storage_changes: changes.storage_changes.into_iter().map(Into::into).collect(),
            storage_reads: changes.storage_reads.into_iter().map(B256::from).collect(),
            balance_changes: changes.balance_changes.into_iter().map(Into::into).collect(),
            nonce_changes: changes.nonce_changes.into_iter().map(Into::into).collect(),
            code_changes: changes.code_changes.into_iter().map(Into::into).collect(),
        }
    }
}

/// All writes to a single storage slot.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SlotChanges {
    /// The storage slot key as a padded 32-byte word.
    pub key: B256,
    /// The writes to the slot, in access order.
    pub changes: Vec<StorageChange>,
}

impl From<alloy_eip7928::SlotChanges> for SlotChanges {
    fn from(changes: alloy_eip7928::SlotChanges) -> Self {
        Self {
            key: B256::from(changes.slot),
            changes: changes.changes.into_iter().map(Into::into).collect(),
        }
    }
}

/// A single write to a storage slot.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StorageChange {
    /// The block access index at which the write occurred.
    pub index: U64,
    /// The value written to the slot as a padded 32-byte word.
    pub value: B256,
}

impl From<alloy_eip7928::StorageChange> for StorageChange {
    fn from(change: alloy_eip7928::StorageChange) -> Self {
        Self { index: U64::from(change.block_access_index.0), value: B256::from(change.new_value) }
    }
}

/// A single balance change of an account.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BalanceChange {
    /// The block access index at which the change occurred.
    pub index: U64,
    /// The post-balance of the account.
    pub value: U256,
}

impl From<alloy_eip7928::BalanceChange> for BalanceChange {
    fn from(change: alloy_eip7928::BalanceChange) -> Self {
        Self { index: U64::from(change.block_access_index.0), value: change.post_balance }
    }
}

/// A single nonce change of an account.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NonceChange {
    /// The block access index at which the change occurred.
    pub index: U64,
    /// The new nonce of the account.
    pub value: U64,
}

impl From<alloy_eip7928::NonceChange> for NonceChange {
    fn from(change: alloy_eip7928::NonceChange) -> Self {
        Self { index: U64::from(change.block_access_index.0), value: U64::from(change.new_nonce) }
    }
}

/// A single code change of an account.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CodeChange {
    /// The block access index at which the change occurred.
    pub index: U64,
    /// The new code of the account.
    pub code: Bytes,
}

impl From<alloy_eip7928::CodeChange> for CodeChange {
    fn from(change: alloy_eip7928::CodeChange) -> Self {
        Self { index: U64::from(change.block_access_index.0), code: change.new_code }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy_eip7928::BlockAccessIndex;
    use alloy_primitives::{address, bytes};

    #[test]
    fn account_access_serializes_to_execution_apis_schema() {
        let account = alloy_eip7928::AccountChanges {
            address: address!("0x2adc25665018aa1fe0e6bc666dac8fc2697ff9ba"),
            storage_changes: vec![alloy_eip7928::SlotChanges {
                slot: U256::from(1),
                changes: vec![alloy_eip7928::StorageChange {
                    block_access_index: BlockAccessIndex(0),
                    new_value: U256::from(2),
                }],
            }],
            storage_reads: vec![U256::from(3)],
            balance_changes: vec![alloy_eip7928::BalanceChange {
                block_access_index: BlockAccessIndex(1),
                post_balance: U256::from(1000),
            }],
            nonce_changes: vec![alloy_eip7928::NonceChange {
                block_access_index: BlockAccessIndex(2),
                new_nonce: 5,
            }],
            code_changes: vec![alloy_eip7928::CodeChange {
                block_access_index: BlockAccessIndex(3),
                new_code: bytes!("0x60006000"),
            }],
        };

        let json = serde_json::to_value(AccountAccess::from(account)).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "address": "0x2adc25665018aa1fe0e6bc666dac8fc2697ff9ba",
                "storageChanges": [{
                    "key": "0x0000000000000000000000000000000000000000000000000000000000000001",
                    "changes": [{
                        "index": "0x0",
                        "value": "0x0000000000000000000000000000000000000000000000000000000000000002",
                    }],
                }],
                "storageReads": [
                    "0x0000000000000000000000000000000000000000000000000000000000000003",
                ],
                "balanceChanges": [{ "index": "0x1", "value": "0x3e8" }],
                "nonceChanges": [{ "index": "0x2", "value": "0x5" }],
                "codeChanges": [{ "index": "0x3", "code": "0x60006000" }],
            })
        );
    }

    #[test]
    fn empty_account_access_serializes_all_fields() {
        let account = AccountAccess::from(alloy_eip7928::AccountChanges::default());
        let json = serde_json::to_value(account).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "address": "0x0000000000000000000000000000000000000000",
                "storageChanges": [],
                "storageReads": [],
                "balanceChanges": [],
                "nonceChanges": [],
                "codeChanges": [],
            })
        );
    }
}
