//! Executor metrics.
//!
//! Block processing related to syncing should take care to update the metrics by using either
//! [`ExecutorMetrics::execute_metered`] or [`ExecutorMetrics::metered_one`].
use crate::{execute::Executor, Database, OnStateHook};
use alloy_consensus::BlockHeader;
use alloy_evm::block::StateChangeSource;
use metrics::{Counter, Gauge, Histogram, describe_histogram};
use reth_execution_types::BlockExecutionOutput;
use reth_metrics::Metrics;
use reth_primitives_traits::{NodePrimitives, RecoveredBlock};
use revm::state::EvmState;
use std::time::Instant;
use once_cell::sync::Lazy;

/// Wrapper struct that combines metrics and state hook
struct MeteredStateHook {
    metrics: ExecutorMetrics,
    inner_hook: Box<dyn OnStateHook>,
}

impl OnStateHook for MeteredStateHook {
    fn on_state(&mut self, source: StateChangeSource, state: &EvmState) {
        // Update the metrics for the number of accounts, storage slots and bytecodes loaded
        let accounts = state.keys().len();
        let storage_slots = state.values().map(|account| account.storage.len()).sum::<usize>();
        let bytecodes = state
            .values()
            .filter(|account| !account.info.is_empty_code_hash())
            .collect::<Vec<_>>()
            .len();

        self.metrics.accounts_loaded_histogram.record(accounts as f64);
        self.metrics.storage_slots_loaded_histogram.record(storage_slots as f64);
        self.metrics.bytecodes_loaded_histogram.record(bytecodes as f64);

        // Call the original state hook
        self.inner_hook.on_state(source, state);
    }
}

/// Executor metrics.
// TODO(onbjerg): add sload/sstore
#[derive(Metrics, Clone)]
#[metrics(scope = "sync.execution")]
pub struct ExecutorMetrics {
    /// The total amount of gas processed.
    pub gas_processed_total: Counter,
    /// The instantaneous amount of gas processed per second.
    pub gas_per_second: Gauge,

    /// The Histogram for amount of time taken to execute blocks.
    pub execution_histogram: Histogram,
    /// The total amount of time it took to execute the latest block.
    pub execution_duration: Gauge,

    /// The Histogram for number of accounts loaded when executing the latest block.
    pub accounts_loaded_histogram: Histogram,
    /// The Histogram for number of storage slots loaded when executing the latest block.
    pub storage_slots_loaded_histogram: Histogram,
    /// The Histogram for number of bytecodes loaded when executing the latest block.
    pub bytecodes_loaded_histogram: Histogram,

    /// The Histogram for number of accounts updated when executing the latest block.
    pub accounts_updated_histogram: Histogram,
    /// The Histogram for number of storage slots updated when executing the latest block.
    pub storage_slots_updated_histogram: Histogram,
    /// The Histogram for number of bytecodes updated when executing the latest block.
    pub bytecodes_updated_histogram: Histogram,
}

impl ExecutorMetrics {
    fn metered<F, R, B>(&self, block: &RecoveredBlock<B>, f: F) -> R
    where
        F: FnOnce() -> R,
        B: reth_primitives_traits::Block,
    {
        // Execute the block and record the elapsed time.
        let execute_start = Instant::now();
        let output = f();
        let execution_duration = execute_start.elapsed().as_secs_f64();

        // Update gas metrics.
        self.gas_processed_total.increment(block.header().gas_used());
        self.gas_per_second.set(block.header().gas_used() as f64 / execution_duration);
        self.execution_histogram.record(execution_duration);
        self.execution_duration.set(execution_duration);

        output
    }

    /// Execute the given block using the provided [`Executor`] and update metrics for the
    /// execution.
    ///
    /// Compared to [`Self::metered_one`], this method additionally updates metrics for the number
    /// of accounts, storage slots and bytecodes loaded and updated.
    /// Execute the given block using the provided [`Executor`] and update metrics for the
    /// execution.
    pub fn execute_metered<E, DB>(
        &self,
        executor: E,
        input: &RecoveredBlock<<E::Primitives as NodePrimitives>::Block>,
        state_hook: Box<dyn OnStateHook>,
    ) -> Result<BlockExecutionOutput<<E::Primitives as NodePrimitives>::Receipt>, E::Error>
    where
        DB: Database,
        E: Executor<DB>,
    {
        // clone here is cheap, all the metrics are Option<Arc<_>>. additionally
        // they are gloally registered so that the data recorded in the hook will
        // be accessible.
        let wrapper = MeteredStateHook { metrics: self.clone(), inner_hook: state_hook };

        // Use metered to execute and track timing/gas metrics
        let output = self.metered(input, || executor.execute_with_state_hook(input, wrapper))?;

        // Update the metrics for the number of accounts, storage slots and bytecodes updated
        let accounts = output.state.state.len();
        let storage_slots =
            output.state.state.values().map(|account| account.storage.len()).sum::<usize>();
        let bytecodes = output.state.contracts.len();

        self.accounts_updated_histogram.record(accounts as f64);
        self.storage_slots_updated_histogram.record(storage_slots as f64);
        self.bytecodes_updated_histogram.record(bytecodes as f64);

        Ok(output)
    }

    /// Execute the given block and update metrics for the execution.
    pub fn metered_one<F, R, B>(&self, input: &RecoveredBlock<B>, f: F) -> R
    where
        F: FnOnce(&RecoveredBlock<B>) -> R,
        B: reth_primitives_traits::Block,
    {
        self.metered(input, || f(input))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy_eips::eip7685::Requests;
    use alloy_primitives::{B256, U256};
    use metrics_util::debugging::{DebugValue, DebuggingRecorder, Snapshotter};
    use reth_ethereum_primitives::EthPrimitives;
    use reth_execution_types::BlockExecutionResult;
    use revm::{
        database::State,
        database_interface::EmptyDB,
        state::{Account, AccountInfo, AccountStatus, EvmStorage, EvmStorageSlot},
    };
    use std::sync::mpsc;

    /// A mock executor that simulates state changes
    struct MockExecutor {
        state: EvmState,
    }

    impl<DB: Database + Default> Executor<DB> for MockExecutor {
        type Primitives = EthPrimitives;
        type Error = std::convert::Infallible;

        fn execute_one(
            &mut self,
            _block: &RecoveredBlock<<Self::Primitives as NodePrimitives>::Block>,
        ) -> Result<BlockExecutionResult<<Self::Primitives as NodePrimitives>::Receipt>, Self::Error>
        {
            Ok(BlockExecutionResult {
                receipts: vec![],
                requests: Requests::default(),
                gas_used: 0,
            })
        }

        fn execute_one_with_state_hook<F>(
            &mut self,
            _block: &RecoveredBlock<<Self::Primitives as NodePrimitives>::Block>,
            mut hook: F,
        ) -> Result<BlockExecutionResult<<Self::Primitives as NodePrimitives>::Receipt>, Self::Error>
        where
            F: OnStateHook + 'static,
        {
            // Call hook with our mock state
            hook.on_state(StateChangeSource::Transaction(0), &self.state);

            Ok(BlockExecutionResult {
                receipts: vec![],
                requests: Requests::default(),
                gas_used: 0,
            })
        }

        fn into_state(self) -> revm::database::State<DB> {
            State::builder().with_database(Default::default()).build()
        }

        fn size_hint(&self) -> usize {
            0
        }
    }

    struct ChannelStateHook {
        output: i32,
        sender: mpsc::Sender<i32>,
    }

    impl OnStateHook for ChannelStateHook {
        fn on_state(&mut self, _source: StateChangeSource, _state: &EvmState) {
            let _ = self.sender.send(self.output);
        }
    }

    /// Helper to set up a test metrics recorder
    fn setup_test_recorder() -> Snapshotter {
        static RECORDER: Lazy<Snapshotter> = Lazy::new(|| {
            let recorder = DebuggingRecorder::new();
            let snapshotter = recorder.snapshotter();
            let _ = recorder.install();
            snapshotter
        });
        
        RECORDER.clone()
    }

    #[test]
    fn test_executor_metrics_hook_metrics_recorded() {
        let snapshotter = setup_test_recorder();
        let metrics = ExecutorMetrics::default();
        let input = RecoveredBlock::default();

        let (tx, _rx) = mpsc::channel();
        let expected_output = 42;
        let state_hook = Box::new(ChannelStateHook { sender: tx, output: expected_output });

        let state = {
            let mut state = EvmState::default();
            let storage =
                EvmStorage::from_iter([(U256::from(1), EvmStorageSlot::new(U256::from(2)))]);
            state.insert(
                Default::default(),
                Account {
                    info: AccountInfo {
                        balance: U256::from(100),
                        nonce: 10,
                        code_hash: B256::ZERO,
                        code: Default::default(),
                    },
                    storage,
                    status: AccountStatus::Loaded,
                },
            );
            state
        };
        let executor = MockExecutor { state };
        let _result = metrics.execute_metered::<_, EmptyDB>(executor, &input, state_hook).unwrap();

        let snapshot = snapshotter.snapshot().into_vec();

        for metric in snapshot {
            let metric_name = metric.0.key().name();
            if metric_name == "sync.execution.accounts_loaded_histogram" ||
                metric_name == "sync.execution.storage_slots_loaded_histogram" ||
                metric_name == "sync.execution.bytecodes_loaded_histogram"
            {
                if let DebugValue::Histogram(vs) = metric.3 {
                    assert!(
                        vs.iter().any(|v| v.into_inner() > 0.0),
                        "metric {metric_name} not recorded"
                    );
                }
            }
        }
    }

    #[test]
    fn test_executor_metrics_hook_called() {
        let metrics = ExecutorMetrics::default();
        let input = RecoveredBlock::default();

        let (tx, rx) = mpsc::channel();
        let expected_output = 42;
        let state_hook = Box::new(ChannelStateHook { sender: tx, output: expected_output });

        let state = EvmState::default();

        let executor = MockExecutor { state };
        let _result = metrics.execute_metered::<_, EmptyDB>(executor, &input, state_hook).unwrap();

        let actual_output = rx.try_recv().unwrap();
        assert_eq!(actual_output, expected_output);
    }
}

/// Metrics for EVM execution.
#[derive(Metrics)]
#[metrics(scope = "evm")]
pub struct EvmMetrics {
    /// Histogram measuring transaction execution time in milliseconds.
    pub transaction_execution_time: Histogram,
}

/// Get a reference to the EVM metrics singleton.
pub fn evm_metrics() -> &'static EvmMetrics {
    static METRICS: Lazy<EvmMetrics> = Lazy::new(|| {
        // Describe the histogram for better documentation in metrics output
        describe_histogram!(
            "evm.transaction_execution_time", 
            "Time taken to execute a single transaction in milliseconds"
        );
        
        // Create the metrics instance - the Metrics derive macro handles registration
        EvmMetrics::default()
    });
    &METRICS
}

/// Helper struct to measure and record transaction execution time.
#[derive(Debug)]
pub struct TimingHelper {
    start: std::time::Instant,
}

impl TimingHelper {
    /// Create a new timing helper and start the timer.
    pub fn start() -> Self {
        Self {
            start: std::time::Instant::now(),
        }
    }

    /// Record the elapsed time to metrics and return the elapsed time in milliseconds.
    pub fn stop(self) -> f64 {
        let elapsed = self.start.elapsed();
        let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
        
        // Record to metrics
        evm_metrics().transaction_execution_time.record(elapsed_ms);
        
        elapsed_ms
    }
}

#[cfg(test)]
mod evm_metrics_tests {
    use super::*;
    use metrics_util::debugging::{DebuggingRecorder, Snapshotter};
    use std::time::Duration;
    
    /// Helper to set up a test metrics recorder
    fn setup_test_recorder() -> Snapshotter {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        let _ = recorder.install();
        snapshotter
    }
    
    #[test]
    fn test_metrics_creation() {
        // Just make sure metrics can be created without errors
        let _metrics = evm_metrics();
    }
    
    #[test]
    fn test_timing_helper() {
        // Set up metrics recording
        let _recorder = setup_test_recorder();
        
        // Create a timing helper
        let timer = TimingHelper::start();
        
        // Simulate some work
        std::thread::sleep(Duration::from_millis(10));
        
        // Record the timing
        let elapsed_ms = timer.stop();
        
        // Verify timing is reasonable
        assert!(elapsed_ms >= 5.0, "Elapsed time should be at least 5ms, got {elapsed_ms}ms");
    }
}
