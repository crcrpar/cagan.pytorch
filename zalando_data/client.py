"""Client utilities."""
import json

import requests


def get_articles(zalandoquery, language='nl-NL', clientname="rolandisshopping"):
    """Client name does not really matter, but you have to fill it in.

    Note that as I am located in the Netherlands all my code is based on the Dutch Zalando API.
    """

    url = zalandoquery.build_query()
    headers = {
        'accept-language': language,
        'x-client-name': clientname,
    }

    response = requests.request("GET", url, headers=headers)
    return json.loads(response.text)


def get_possible_category_keys(language='nl-NL', clientname="rolandisshopping"):
    """Returns possible category keys you can use to filter results.

    Client name does not really matter, but you have to fill it in.
    Note that as I am located in the Netherlands all my code is based on the Dutch Zalando API.
    """

    baseurl = "https://api.zalando.com/categories"
    headers = {
        'accept-language': language,
        'x-client-name': clientname,
    }
    response = requests.request("GET", baseurl, headers=headers)
    pages = json.loads(response.text)['totalPages']
    allcategories = []

    for page in range(1, pages + 1):  # 1-indexed pages
        url = baseurl + "?page=" + str(page)
        response = requests.request("GET", url, headers=headers)

        for categorycollection in json.loads(response.text)['content']:
            allcategories.append(categorycollection['key'])
    return allcategories


def get_articles_all_pages(querybuilder):
    json_data = get_articles(querybuilder)
    if 'totalPages' not in json_data:
        raise ValueError("Error, response Zalando: " + str(json_data))
    page_count = json_data['totalPages']
    for page in range(1, page_count + 1):
        querybuilder.set_page(page)
        json_data = get_articles(querybuilder)

        for item in json_data['content']:
            yield item


def get_url(item):
    return item['shopUrl']


def size_in_stock(item, size_str):
    """Example: has_size(item, '32x36')."""
    for unit in item['units']:
        if unit['size'] == size_str and unit['stock'] > 0:
            return True
    return False


def has_attribute(item, attributename, attributevalue):
    for attribute in item['attributes']:
        if attribute['name'] == attributename and attributevalue in attribute['values']:
            return True
    return False


def get_attribute(item, attributename):
    for attribute in item['attributes']:
        if attribute['name'] == attributename:
            return attribute['values']
    return None


def get_image_urls(item):
    imageurl = []
    for image in item['media']['images']:
        imageurl.append(image['smallHdUrl'])
    return imageurl


def is_denim(item):
    return has_attribute(item, 'materiaalverwerking', 'denim')


def is_slim_fit(item):
    return has_attribute(item, 'Pasvorm', 'slim fit')


def get_the_price(item, size=None):
    if not size:
        firstunit = item['units'][0]
        return firstunit['price']['value'], firstunit['originalPrice']['value']
    for unit in item['units']:
        if unit['size'] == size:
            return unit['price']['value'], unit['originalPrice']['value']
