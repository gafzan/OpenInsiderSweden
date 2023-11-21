"""soup_utils.py"""

import lxml
from bs4 import BeautifulSoup


def get_number_of_pages(page_content: {bytes, str}) -> int:
    """
    Returns the max number of pages as an int
    :param page_content: bytes or str
    :return: int
    """
    soup = BeautifulSoup(page_content, "lxml")
    row = soup.find("div", class_="pager")
    max_page_num_str = row.find_all('li')[-2].text  # get the element left of the 'Next >' button
    max_page_num = int(max_page_num_str.split('p')[-1])
    return max_page_num


def get_trade_entries_from_page(page_content: bytes) -> list:
    """
    Extract the result_l from an HTLM table from a web page and return the result as list
    :param page_content: bytes
    :return: list or tuple
    """
    soup = BeautifulSoup(page_content, "lxml")
    table = soup.find("table")
    rows = table.find_all("tr")
    rows.pop(0)
    data = [[col.get_text(strip=True) for col in row.find_all("td")] for row in rows]
    return data
