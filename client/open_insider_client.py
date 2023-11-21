"""open_insider_client.py"""

import re
import pandas as pd
import numpy as np

import logging

from swedish_market_insights.inside_trades import InsideTradesAPI

# TODO this mapping is not completed yet...
corp_title_map = {
    'Officer':
        [
            'Chief Executive Officer (CEO)',
            'Deputy CEO',
            'Chief Financial Officer (CFO)',
            'Chief Operating Officer (COO)',
            'Chief Operations Officer',
            'Chairman of the Board of Directors',
            'Other senior executive'
        ],
    'Non-officer':
        {
            'Directors':
                [
                    'Member of the Board of Directors',
                    'Deputy Member of the Board of Directors',
                    'Styrelseledamot/suppleant'
                ],
            'Other':
                [
                    "Other member of the company's administrative, management or supervisory body",
                    "Employee Representative of the Board of Directors or deputy Employee Representative of the Board of Directors",
                    "Managing Directory",
                    "Deputy Managing Director",
                ]
        }
}

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


class OpenInsiderClient:

    def __init__(self, data: {str, pd.DataFrame, None}=None):
        self.insider_trades_api = InsideTradesAPI()
        self.data = data
        self.filtered_data = None

    def get_data(self):
        if self.data is None:
            logger.debug('Donwload insider trades looking back 3 months')
            # download insider trades done for the past 3 months
            self.data = self.insider_trades_api.get_trades_published_months_ago(months_ago=3)
            return self.data
        elif isinstance(self.data, str):
            logger.debug(f'Reads data saved in {self.data}')
            return pd.read_csv(self.data, index_col=0)
        else:
            return self.data

    def clean_data(self, only_acquisitions: bool = False) -> None:
        """
        Sets filtered_data after filtering column values to only have buy & sell, no status, a defined ISIN, only
        shares and currency is SEK.
        Convert publication and transaction dates to datetime.
        Format name of person to be Camel Case
        :param only_acquisitions: bool if True only includes buys, else includes buys and sells
        :return: None
        """
        logger.debug("cleans data")
        if only_acquisitions:
            eligible_transactions = ['Acquisition']
        else:
            eligible_transactions = ['Acquisition', 'Disposal']
        clean_df = self.filtered_data[self.filtered_data['Nature of transaction'].isin(eligible_transactions)
                      & ((self.filtered_data['Status'] == '')
                         | self.filtered_data['Status'].isnull())
                      & ((self.filtered_data['ISIN'] != '')
                         | self.filtered_data['ISIN'] != np.nan)
                      & (self.filtered_data['Instrument type'] == 'Share')
                      & (self.filtered_data['Currency'] == 'SEK')]
        clean_df.loc[:, 'Publication date'] = pd.to_datetime(clean_df['Publication date'].copy(), format="%d/%m/%Y")
        clean_df.loc[:, 'Transaction date'] = pd.to_datetime(clean_df['Transaction date'].copy(), format="%d/%m/%Y")
        clean_df.loc[:, 'Person discharging managerial responsibilities'] = (
            clean_df.loc[:, 'Person discharging managerial responsibilities'].str.title()).replace(' ', '', regex=True)
        self.filtered_data = clean_df.copy()
        return

    def add_additional_columns(self) -> None:
        """
        Sets filtered_data after adding Officer bool and Amount
        :return: None
        """
        logger.debug("add columns")
        officers_re_pattern = "|".join([re.escape(e) for e in corp_title_map['Officer']])
        self.filtered_data.loc[:, 'Officer'] = self.filtered_data.Position.str.contains(officers_re_pattern)
        self.filtered_data.loc[:, 'Amount'] = self.filtered_data['Price'].copy() * self.filtered_data['Volume'].copy()
        return

    def filter_by_position(self, valid_positions: list) -> None:
        """
        Sets filtered_data after filtering 'Position' column to have values to be included whole or as a substring in
        the specified list.
        E.g. valid_positions = ['CEO'] will filter column values such as 'Chief Executive Officer (CEO)/Managing Directory'
        and 'CEO'
        :param valid_positions: list of str
        :return: DataFrame
        """
        logger.debug("filter by positions")
        positions = self.filtered_data.Position
        valid_positions = [re.escape(e) for e in valid_positions.copy()]  # take care of weird characters
        re_pattern = "|".join(valid_positions)
        self.filtered_data = self.filtered_data.loc[positions.str.contains(re_pattern)]
        return

    def filter_by_large_trades(self, min_amount_per_insider_sek: float):
        """
        Filter out trades during the period that are smaller than a specified amount made by a particular insider.
        Note that if an insider buys 5 times each of an amount equal to SEK 10,000 and the minimum amount is set to
        SEK 40,000 the trades will be included since the accumulated Amount is SEK 50,000. This is to not exclude
        trades done using smoothing i.e. spread a large trade over a period of days
        :param min_amount_per_insider_sek: float
        :return:
        """
        logger.debug(f'filter out insiders who has traded less than SEK {min_amount_per_insider_sek} during the period')
        # filter out small trades done by insiders during the entire period i.e. not just during one day
        # create a pivot table with ISIN as index, persons as column headers and the Amount sum for the period as data
        amount_per_person_isin_idx_df = self.filtered_data.pivot_table(values='Amount',
                                                                       index='ISIN', columns='Person discharging managerial responsibilities',
                                                       aggfunc='sum').fillna(0)
        # 1 if large enough trade, else 0
        large_trades_indi_df = pd.DataFrame(
            data=np.where(amount_per_person_isin_idx_df >= min_amount_per_insider_sek, 1, 0),
            columns=amount_per_person_isin_idx_df.columns,
            index=amount_per_person_isin_idx_df.index)
        # drop the rows and columns without nay large amounts i.e. indicator is 0 and store the remaining rows and cols
        filtered_large_trades_indi_df = large_trades_indi_df.loc[(large_trades_indi_df.sum(axis=1) != 0),
                                                                 (large_trades_indi_df.sum(axis=0) != 0)]
        eligible_isin = list(filtered_large_trades_indi_df.index)
        eligible_person = list(filtered_large_trades_indi_df.columns)

        # filter out based on the eligible ISINs and persons
        self.filtered_data = self.filtered_data[
            self.filtered_data['ISIN'].isin(eligible_isin)
            & self.filtered_data['Person discharging managerial responsibilities'].isin(eligible_person)
        ].copy()
        return

    def get_num_insiders_per_isin_df(self) -> pd.DataFrame:
        """
        Returns a DataFrame with ISIN as index and number of insiders with trades during the period as column
        :return: DataFrame
        """
        return self.filtered_data.groupby('ISIN')['Person discharging managerial responsibilities'].nunique().to_frame(
            'Insiders')

    def get_num_officers_per_isin_df(self) -> pd.DataFrame:
        """
        Returns a DataFrame with ISIN as index and number of Officers with insider trades during the period as column
        :return: DataFrame
        """
        officer_count_per_isin = self.filtered_data.pivot_table(values='Officer', index='ISIN',
                                              columns='Person discharging managerial responsibilities',
                                              aggfunc='sum').fillna(0)
        officer_count = pd.DataFrame(data=np.sum(np.where(officer_count_per_isin > 0, 1, 0), axis=1),
                                     index=officer_count_per_isin.index,
                                     columns=['Officers'])
        return officer_count

    def get_cluster_buys(self, min_amount_ksek: float = 1_000_000, min_amount_per_insider_sek: float = 100_000.0,
                         min_num_insiders: int = 2, min_num_officers: int = 0) -> pd.DataFrame:
        """
        Returns a DataFrame with ISIN as index and aggregated data columns such as Amount, number of Insiders and
        Officers. Results are sorted by Publication date, number of Insiders and Amount
        :param min_amount_ksek: float minimum total Amount traded per ISIN
        :param min_amount_per_insider_sek: float minimum amount traded per insider during the period
        :param min_num_insiders: int
        :param min_num_officers: int
        :return: DataFrame
        """
        # initialize the filtered data, clean it, add additional columns and filter out small trades per insider
        self.filtered_data = self.get_data()  # initialize to the raw data
        self.clean_data(only_acquisitions=True)
        self.add_additional_columns()
        self.filter_by_large_trades(min_amount_per_insider_sek=min_amount_per_insider_sek)

        # group the result by ISIN aggregating dates as the latest one available, sum Volumes and Amounts and add a
        # Volume weighted average price
        col_agg_mthd_map = {'Publication date': 'max', 'Transaction date': 'max', 'Volume': 'sum', 'Amount': 'sum'}
        isin_grouped_df = self.filtered_data.groupby('ISIN')[['Publication date', 'Transaction date', 'Volume',
                                                              'Amount']].agg(col_agg_mthd_map).join(
            self.filtered_data.groupby('ISIN').apply(
                lambda x: pd.Series(np.average(x[["Price"]], weights=x["Volume"], axis=0), ["Avg. Price"])))

        # add column counting insiders
        isin_grouped_df = isin_grouped_df.join(self.get_num_insiders_per_isin_df())

        # add column counting officers
        isin_grouped_df = isin_grouped_df.join(self.get_num_officers_per_isin_df())

        # filter based on size of total amount traded, number of insiders and officers
        isin_grouped_df = isin_grouped_df[
            (isin_grouped_df.Amount >= min_amount_ksek)
            & (isin_grouped_df.Insiders >= min_num_insiders)
            & (isin_grouped_df.Officers >= min_num_officers)
        ].copy()

        # sort by publication date, number of insiders and amount
        isin_grouped_df.sort_values(['Publication date', 'Insiders', 'Amount'], inplace=True,
                                    ascending=[False, False, False])
        isin_grouped_df.reset_index(inplace=True)

        # add ticker, name, sector, industry to the result
        stock_info_df = pd.read_csv('swedish_stocks.csv')
        result = pd.merge(isin_grouped_df, stock_info_df, how='left', on='ISIN')
        result.set_index('ISIN', inplace=True)
        return result

