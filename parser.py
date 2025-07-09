#!/usr/bin/env python3
"""
Continuous parser for Reth node logs (rotated '.1'–'.N' plus current 'reth.log').
Computes IO_time and EVM_time per tx and uploads Parquet files to GCS,
flushing per block number (rather than fixed chunk sizes).

Requirements:
  pip install pandas google-cloud-storage pyarrow
"""

import os
import re
import json
import time
import glob
import argparse
import tempfile
import logging
import signal
import atexit

import pandas as pd
from google.cloud import storage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Regex patterns for parsing
# New patterns for your timing instrumentation
PATTERN_BASIC_ACCOUNT = re.compile(r'basic_account_info(?:_ref)? for (0x[a-fA-F0-9]+) took (\d+) microseconds')
PATTERN_STORAGE = re.compile(r'storage(?:_ref)? for (0x[a-fA-F0-9]+) at index (0x[a-fA-F0-9]+) took (\d+) microseconds')
PATTERN_CODE_HASH = re.compile(r'code_by_hash(?:_ref)? for (0x[a-fA-F0-9]+) took (\d+) microseconds')
PATTERN_BLOCK_HASH = re.compile(r'block_hash(?:_ref)? for block (\d+) took (\d+) microseconds')

# New pattern for EVM execution timing
PATTERN_EVM_EXEC = re.compile(r'Finished executing transaction tx_index=(\d+) block=(\d+) time=(\d+) microseconds')

# Keep original patterns as fallback
PATTERN_ACCESS = re.compile(r'INFO\s+.*?:\s+(?:Account accessed|storage accessed|code hash accessed).*?time=(\d+\.?\d*)(ms|µs)')
PATTERN_EVM    = re.compile(r'INFO\s+.*?:\s+Finished executing evm\s+time=(\d+\.?\d*)(ms|µs)')
PATTERN_TX     = re.compile(r'INFO\s+.*?:\s+Finished processing tx\s+tx_hash=(\S+)\s+time=(\d+\.?\d*)(ms|µs)')
# Updated block pattern to capture block number from lines like:
# "... Finished inserting block block=NumHash { number: 22323394, hash: ... }"
PATTERN_BLOCK  = re.compile(r'.*Finished inserting block.*?number\s*:\s*(\d+)')

# Default parser state keys
STATE_DEFAULT = {
    'inode': None,
    'offset': 0,
    'uploaded_blocks': []  # Track uploaded block numbers
}

# Default location for uploaded blocks file
UPLOADED_BLOCKS_FILE = '/tmp/reth_uploaded_blocks.json'

# Convert time unit to ms
def parse_time(val: str, unit: str) -> float:
    x = float(val)
    return x / 1000.0 if unit == 'µs' else x

# Load uploaded blocks from tmp file
def load_uploaded_blocks(path: str = UPLOADED_BLOCKS_FILE) -> list:
    try:
        with open(path, 'r') as f:
            blocks = json.load(f)
        if isinstance(blocks, list):
            logger.info(f"Loaded {len(blocks)} uploaded blocks from {path}")
            return blocks
    except Exception as e:
        logger.info(f"No uploaded blocks file found or error loading: {e}")
    return []

# Save uploaded blocks to tmp file
def save_uploaded_blocks(blocks: list, path: str = UPLOADED_BLOCKS_FILE, max_blocks: int = 10000):
    try:
        # Keep only the most recent blocks
        if len(blocks) > max_blocks:
            blocks = sorted(blocks)[-max_blocks:]
        
        tmp = f"{path}.tmp"
        with open(tmp, 'w') as f:
            json.dump(blocks, f)
        os.replace(tmp, path)
        logger.info(f"Saved {len(blocks)} uploaded blocks to {path}")
    except Exception as e:
        logger.error(f"Failed to save uploaded blocks: {e}")

# Save state to disk
def save_state(path: str, state: dict):
    tmp = f"{path}.tmp"
    with open(tmp, 'w') as f:
        # Save all state including uploaded blocks
        state_to_save = {k: state.get(k, STATE_DEFAULT[k]) for k in STATE_DEFAULT}
        json.dump(state_to_save, f)
    os.replace(tmp, path)
    logger.debug(f"State saved: inode={state['inode']} offset={state['offset']} uploaded_blocks={len(state.get('uploaded_blocks', []))}")

# Load state from disk
def load_state(path: str) -> dict:
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        # Ensure we have all required keys with defaults
        st = STATE_DEFAULT.copy()
        st.update(data)
        # Ensure uploaded_blocks is a list
        if not isinstance(st.get('uploaded_blocks'), list):
            st['uploaded_blocks'] = []
        logger.info(f"Loaded state: inode={st['inode']} offset={st['offset']} uploaded_blocks={len(st['uploaded_blocks'])}")
        return st
    except Exception:
        logger.info("No valid state file, starting fresh.")
        return STATE_DEFAULT.copy()

# Check if blob exists in GCS
def check_blob_exists(bucket: str, blob_name: str) -> bool:
    try:
        client = storage.Client()
        blob = client.bucket(bucket).blob(blob_name)
        return blob.exists()
    except Exception as e:
        logger.warning(f"Error checking blob existence: {e}")
        return False

# Upload DataFrame as Parquet to GCS
def upload_parquet(df: pd.DataFrame, bucket: str, blob_name: str):
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
        df.to_parquet(tmp.name, index=False)
    client = storage.Client()
    client.bucket(bucket).blob(blob_name).upload_from_filename(tmp.name)
    os.remove(tmp.name)
    logger.info(f"Uploaded {blob_name} ({len(df)} rows)")

# Shared state
records = []
current_io = 0.0
current_evm = 0.0
current_tx_index = None

# Process a single log line
def process_line(line: str, state: dict, fh) -> None:
    global records, current_io, current_evm, current_tx_index
    
    # Check for new timing instrumentation patterns first
    # Basic account access
    if (m := PATTERN_BASIC_ACCOUNT.search(line)):
        microseconds = int(m.group(2))
        current_io += microseconds / 1000.0  # Convert to ms
        return
    
    # Storage access
    if (m := PATTERN_STORAGE.search(line)):
        microseconds = int(m.group(3))
        current_io += microseconds / 1000.0  # Convert to ms
        return
    
    # Code hash access
    if (m := PATTERN_CODE_HASH.search(line)):
        microseconds = int(m.group(2))
        current_io += microseconds / 1000.0  # Convert to ms
        return
    
    # Block hash access
    if (m := PATTERN_BLOCK_HASH.search(line)):
        microseconds = int(m.group(2))
        current_io += microseconds / 1000.0  # Convert to ms
        return
    
    # EVM execution time
    if (m := PATTERN_EVM_EXEC.search(line)):
        tx_idx = int(m.group(1))
        block_num = int(m.group(2))
        microseconds = int(m.group(3))
        evm_time = microseconds / 1000.0  # Convert to ms
        
        # The logged time is just EVM execution, not including IO
        # So we store IO time separately and EVM time as-is
        records.append({
            'tx_hash': f'tx_{block_num}_{tx_idx}',  # Temporary identifier
            'IO_time': current_io,
            'EVM_time': evm_time,  # This is pure EVM time, not total
            'block_number': block_num
        })
        
        # Reset counters for next transaction
        current_io = 0.0
        current_tx_index = tx_idx
        
        state['offset'] = fh.tell()
        save_state(state['state_file'], state)
        logger.info(f"Queued tx index {tx_idx} in block {block_num}, buffer size {len(records)}")
        return
    
    # Fallback to original patterns
    # IO events
    if (m := PATTERN_ACCESS.search(line)):
        current_io += parse_time(*m.groups())
        return
    # Raw EVM time
    if (m := PATTERN_EVM.search(line)):
        current_evm = parse_time(*m.groups())
        return
    # Transaction boundary
    if (m := PATTERN_TX.search(line)):
        txh = m.group(1)
        io_t = current_io
        evm_net = current_evm - io_t
        records.append({'tx_hash': txh, 'IO_time': io_t, 'EVM_time': evm_net})
        current_io = current_evm = 0.0
        state['offset'] = fh.tell()
        save_state(state['state_file'], state)
        logger.info(f"Queued tx {txh}, buffer size {len(records)}")
        return
    # Block boundary: flush all pending txs for that block
    if (m := PATTERN_BLOCK.search(line)):
        blk = int(m.group(1))
        if records:
            # Check if block was already uploaded
            if blk in state.get('uploaded_blocks', []):
                logger.info(f"Block {blk} already uploaded, skipping {len(records)} txs")
                records.clear()
            else:
                df = pd.DataFrame(records)
                df['block_number'] = blk
                blob = f"{blk}.parquet"
                
                # Double-check in GCS before uploading
                if check_blob_exists(state['bucket'], blob):
                    logger.info(f"Block {blk} exists in GCS, skipping upload")
                    # Add to uploaded_blocks to avoid future checks
                    if 'uploaded_blocks' not in state:
                        state['uploaded_blocks'] = []
                    state['uploaded_blocks'].append(blk)
                    # Save to tmp file
                    save_uploaded_blocks(state['uploaded_blocks'], max_blocks=state.get('max_tracked_blocks', 10000))
                else:
                    upload_parquet(df, state['bucket'], blob)
                    logger.info(f"Flushed {len(df)} txs for block {blk}")
                    # Track this block as uploaded
                    if 'uploaded_blocks' not in state:
                        state['uploaded_blocks'] = []
                    state['uploaded_blocks'].append(blk)
                    
                    # Trim the list if it gets too large (keep most recent blocks)
                    max_blocks = state.get('max_tracked_blocks', 10000)
                    if len(state['uploaded_blocks']) > max_blocks:
                        # Keep the most recent blocks
                        state['uploaded_blocks'] = sorted(state['uploaded_blocks'])[-max_blocks:]
                        logger.info(f"Trimmed uploaded_blocks list to {max_blocks} most recent blocks")
                    
                    # Save uploaded blocks to tmp file periodically
                    save_uploaded_blocks(state['uploaded_blocks'], max_blocks=max_blocks)
                
                records.clear()
        state['offset'] = fh.tell()
        save_state(state['state_file'], state)

# Process rotated logs (backfill)
# Process rotated logs (backfill)  
def process_rotated_logs(path: str, state: dict):
    base = os.path.basename(path)
    dirp = os.path.dirname(path)
    files = [(int(fn.rsplit('.',1)[-1]), fn)
             for fn in glob.glob(f"{dirp}/{base}.*") if fn.rsplit('.',1)[-1].isdigit()]
    for _, fn in sorted(files, reverse=True):
        logger.info(f"Backfilling {fn}")
        with open(fn, 'r') as f:
            # Use readline() loop so fh.tell() remains available
            while True:
                line = f.readline()
                if not line:
                    break
                process_line(line, state, f)
    # no return needed f, ino

# Open the current log for tailing
def open_log(path: str, state: dict):
    st = os.stat(path)
    ino = st.st_ino
    f = open(path, 'r')
    if state['inode'] == ino:
        f.seek(state['offset'])
        logger.info(f"Resuming at offset {state['offset']}")
    else:
        state['inode'] = ino
        state['offset'] = 0
        logger.info("Starting fresh log tail")
    return f, ino

# Main entrypoint
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--log-file', required=True)
    p.add_argument('--state-file', default=os.path.expanduser('~/.reth_state.json'))
    p.add_argument('--bucket', default='ethereum-execution-times', help='GCS bucket name')
    p.add_argument('--sleep', type=float, default=1.0)
    p.add_argument('--max-tracked-blocks', type=int, default=10000, 
                   help='Maximum number of uploaded blocks to track in memory (default: 10000)')
    args = p.parse_args()

    # Set GCP credentials
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "./google-creds.json"

    state = load_state(args.state_file)
    state.update({
        'state_file': args.state_file, 
        'bucket': args.bucket,
        'max_tracked_blocks': args.max_tracked_blocks
    })
    
    # Load uploaded blocks from tmp file
    uploaded_blocks = load_uploaded_blocks()
    if uploaded_blocks:
        # Merge with existing uploaded blocks in state
        existing = set(state.get('uploaded_blocks', []))
        existing.update(uploaded_blocks)
        state['uploaded_blocks'] = sorted(list(existing))[-args.max_tracked_blocks:]
        logger.info(f"Merged uploaded blocks, total: {len(state['uploaded_blocks'])}")
    
    # Set up exit handlers to save uploaded blocks
    def cleanup():
        logger.info("Saving uploaded blocks before exit...")
        save_uploaded_blocks(state.get('uploaded_blocks', []), max_blocks=args.max_tracked_blocks)
        save_state(state['state_file'], state)
    
    # Register cleanup on exit
    atexit.register(cleanup)
    
    # Handle signals for clean shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        raise KeyboardInterrupt()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Backfill rotated logs on first run
    if state['inode'] is None and state['offset'] == 0:
        process_rotated_logs(os.path.expanduser(args.log_file), state)

    # Tail live log
    fh, ino = open_log(os.path.expanduser(args.log_file), state)
    try:
        while True:
            line = fh.readline()
            if not line:
                state['offset'] = fh.tell()
                save_state(state['state_file'], state)
                time.sleep(args.sleep)
                try:
                    new_ino = os.stat(args.log_file).st_ino
                    if new_ino != ino:
                        fh.close()
                        fh, ino = open_log(os.path.expanduser(args.log_file), state)
                        logger.info("Detected rotation, reopened")
                except FileNotFoundError:
                    time.sleep(args.sleep)
                continue
            process_line(line, state, fh)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        fh.close()
        logger.info("Exit complete")

if __name__ == '__main__':
    main()
