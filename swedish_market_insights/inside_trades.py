"""inside_trades.py"""

from datetime import date
from dateutil.relativedelta import relativedelta
import time
import pandas as pd
import requests
from tqdm import tqdm

from concurrent import futures

from swedish_market_insights.utils import handle_date
from constants import INSIDE_TRADES_BASE_URL
from swedish_market_insights.soup_utils import get_number_of_pages
from swedish_market_insights.soup_utils import get_trade_entries_from_page

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


class InsideTradesAPI:

    def __init__(self, batch_size: int = 50, batch_sleep: int = 3, time_out: int = 5):
        """
        Class calling marknadssok.fi.se to get data for transactions made by insiders in swedish companies
        :param batch_size: int number of Threads are split into batches to avoid TimeoutError
        :param batch_sleep: int seconds waiting between each batch
        :param time_out: int if there is no response after time_out a TimeoutError is raised
        """
        self.batch_size = batch_size
        self.batch_sleep = batch_sleep
        self.time_out = time_out
        self._session = None

    def get_trades(self, issuer: str = None, insider: str = None, transaction_date_from: {date, str}=None,
                   transaction_date_to: {date, str}=None, publication_date_from: {date, str}=None,
                   publication_date_to: {date, str}=None) -> pd.DataFrame:
        """
        Returns a DataFrame after searching for issuer name, insider person name, transaction and publication dates
        :param issuer: str
        :param insider: str
        :param transaction_date_from: str, date
        :param transaction_date_to: str, date
        :param publication_date_from: str, date
        :param publication_date_to: str, date
        :return: DataFrame
        """
        # setup a dictionary with parameters to be part of the URL and remove parameters that are not specified
        params = self._get_params(issuer=issuer,
                                  insider=insider,
                                  transaction_date_from=handle_date(transaction_date_from),
                                  transaction_date_to=handle_date(transaction_date_to),
                                  publication_date_from=handle_date(publication_date_from),
                                  publication_date_to=handle_date(publication_date_to))
        result_df = self._get_trades_from_api(params=params)
        self._format_df(df=result_df)
        return result_df

    def get_trades_published_today(self, issuer: str = None, insider: str = None) -> pd.DataFrame:
        """
        Returns a DataFrame with all the insider trades published today
        :param issuer: str
        :param insider: str
        :return: DataFrame
        """
        return self.get_trades(publication_date_from=date.today(), publication_date_to=date.today(), issuer=issuer,
                               insider=insider)

    def get_trades_published_months_ago(self, months_ago: int, issuer: str = None, insider: str = None):
        """
        Returns a DataFrame with all the insider trades published a specified number of months ago
        :param months_ago: int
        :param issuer: str
        :param insider: str
        :return: DataFrame
        """
        return self.get_trades(publication_date_from=date.today() - relativedelta(months=months_ago),
                               publication_date_to=date.today(), issuer=issuer, insider=insider)

    def get_trades_published_ytd(self, issuer: str = None, insider: str = None):
        """
        Returns a DataFrame with all the insider trades published so far year-to-date
        :param issuer: str
        :param insider: str
        :return: DataFrame
        """
        return self.get_trades(publication_date_from=date(date.today().year, 1, 1), publication_date_to=date.today(),
                               issuer=issuer, insider=insider)

    def _get_trades_from_api(self, params: dict) -> pd.DataFrame:
        """
        Calls marknadssok.fi.se API and return trades done by insiders in a DataFrame based on filter parameters
        Script uses pool of threads to run faster
        :param params: dict with parameters used in the URL
        :return: DataFrame
        """
        self._session = requests.Session()
        # on the first page, get the table content as well as the number of pages to be looped over
        res = self._session.get(INSIDE_TRADES_BASE_URL, params=params)
        max_page_num = get_number_of_pages(res.text)
        data = get_trade_entries_from_page(res.text)  # this will be a list of lists, each list is a row

        with futures.ThreadPoolExecutor(10) as executor:  # pool of threads
            key_args = self._get_kwargs_input_list(max_page_num=max_page_num, params=params)
            data_result = []
            key_args_batches = [key_args[i: i + self.batch_size] for i in range(0, len(key_args), self.batch_size)]
            start = time.time()
            for key_args_batch in tqdm(key_args_batches):
                # executed for each value in key_args and returns the futures.results and not futures object
                results = executor.map(self._get_table_data_list_from_fi, key_args_batch, timeout=self.time_out)

                # returns the results in the order that they where started (without any total slowdown)
                data_result.extend(
                    list(results)
                )
                time.sleep(self.batch_sleep)
            end = time.time()
            logger.info(f'Done with web scraping after {round(end - start, 2)} second(s)')

        # flatten the list of list using list comprehension
        logger.debug(f"Flatten the list of lists using list comprehension")
        data.extend(
            [j for i in data_result for j in i]
        )
        columns = ['Publication date', 'Issuer',
                   'Person discharging managerial responsibilities', 'Position',
                   'Closely associated', 'Nature of transaction', 'Instrument name',
                   'Instrument type', 'ISIN', 'Transaction date', 'Volume', 'Unit', 'Price',
                   'Currency', 'Status', 'Details']
        logger.debug(f"Define and return a DataFrame")
        return pd.DataFrame(data=data, columns=columns)

    @staticmethod
    def _get_kwargs_input_list(max_page_num: int, params: dict) -> list:
        """
        Returns a list of dict with arguments to be assigned to get_table_data_list_from_fi function
        :param max_page_num: int
        :param params: dict
        :return: list of dict
        """
        key_args = []
        # starting from the second page since the data from the 1st page is assumed to be downloaded already
        for p_num in range(2, max_page_num + 1):
            par = params.copy()
            par['page'] = p_num  # add/change 'page' key in the parameters that will be part of the URL
            key_args.append(
                {
                    # '_session': _session,
                    'url': INSIDE_TRADES_BASE_URL,
                    'params': par}
            )
        return key_args

    def _get_table_data_list_from_fi(self, kwargs: dict) -> list:
        """
        Gets website content and return a list of list with data
        :param kwargs: dict
        :return: list
        """
        r = self._session.get(kwargs['url'], params=kwargs['params'])
        logger.debug(f"URL: {r.url}")
        return get_trade_entries_from_page(r.text)

    @staticmethod
    def _format_df(df: pd.DataFrame) -> None:
        """
        Re-formats contents in a specified DataFrame
        Volume and Price columns get's converted to floats from str
        :param df:
        :return:
        """
        df['Volume'] = df['Volume'].str.replace(',', '').astype(float)
        df['Price'] = df['Price'].str.replace(',', '').astype(float)
        return

    @staticmethod
    def _get_params(issuer: str = None, insider: str = None, transaction_date_from: date = None,
                    transaction_date_to: date = None, publication_date_from: date = None,
                    publication_date_to: date = None) -> dict:
        """
        Returns a dictionary with the parameters to be included in the URL
        If no parameters are specified a ValueError will be raised
        :param issuer: str
        :param insider: str
        :param transaction_date_from: date
        :param transaction_date_to: date
        :param publication_date_from: date
        :param publication_date_to: date
        :return: dict
        """
        # setup a dictionary with parameters to be part of the URL and remove parameters that are not specified
        params = {'Utgivare': issuer, insider: 'PersonILedandeStällningNamn',
                  'Transaktionsdatum.From': transaction_date_from,
                  'Transaktionsdatum.To': transaction_date_to, "Publiceringsdatum.From": publication_date_from,
                  "Publiceringsdatum.To": publication_date_to}
        params = {
            key: value for key, value in params.items()
            if value is not None
        }
        if params:
            return params
        else:
            raise ValueError("Need to specify at least one search parameter")


