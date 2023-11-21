"""download_cluster_buys.py"""

import pandas as pd
from datetime import datetime
from pathlib import Path
import os

from client.open_insider_client import OpenInsiderClient

root = Path('data')
if not root.exists():
    os.makedirs(root)

# parameters
MIN_AMOUNT_SEK = 1_000_000
MIN_AMOUNT_PER_INSIDER_SEK = 100_000.0
MIN_NUM_INSIDERS = 2
MIN_NUM_OFFICERS = 0
MONTH_LAG = 1

if __name__ == '__main__':
    # get cluster buys using the API
    file_path = root / f"cluster_buys_{MONTH_LAG}M_{datetime.now().strftime('%Y%m%d')}.xlsx"
    client = OpenInsiderClient()
    # client.data = client.insider_trades_api.get_trades_published_months_ago(months_ago=MONTH_LAG)
    client.data = pd.read_excel(file_path, sheet_name='raw_data', index_col=0)
    cluster_buys_df = client.get_cluster_buys(min_amount_ksek=MIN_AMOUNT_SEK,
                                              min_amount_per_insider_sek=MIN_AMOUNT_PER_INSIDER_SEK,
                                              min_num_insiders=MIN_NUM_INSIDERS,
                                              min_num_officers=MIN_NUM_OFFICERS)
    raw_data = client.data
    filtered_data = client.filtered_data

    # save result in excel
    with pd.ExcelWriter(file_path) as writer:
        cluster_buys_df.to_excel(writer, sheet_name='cluster_buys')
        filtered_data.to_excel(writer, sheet_name='filtered_data')
        raw_data.to_excel(writer, sheet_name='raw_data')

