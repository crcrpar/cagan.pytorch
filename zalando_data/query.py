class Query(object):

    def __init__(self, *args, **kwargs):
        """Query for zalando.se API.

        All possible parameters is found via client.py
        """
        self.__dict__ = locals()
        self.age_group = None
        self.gender = None
        self.length = None
        self.size = None
        self.min_price = None
        self.max_price = None
        self.page = None
        self.page_size = None
        self.categories = None
        self.BASE_URL = "https://api.zalando.com/articles?"

    def set_price_range(self, min_price, max_price):
        self.min_price = min_price
        self.max_price = max_price

    def set_page(self, page):
        self.page = page

    def set_page_size(self, page_size):
        self.page_size = page_size

    def set_size(self, size):
        self.size = size

    def set_categories(self, categories):
        if not isinstance(categories, list):
            categories = [categories]
        self.categories = categories

    def set_length(self, length):
        self.length = length

    def set_age_group(self, age_group):
        self.age_group = age_group

    def seg_gender(self, gender):
        self.gender = gender

    def build_query(self):
        to_build = list()

        if self.age_group:
            to_build.append("ageGroup={}".format(self.age_group))
        if self.gender:
            to_build.append("gender={}".format(self.gender))
        if self.length:
            to_build.append("length={}".format(self.length))
        if self.size:
            to_build.append("size={}".format(self.size))
        if self.min_price or self.max_price:
            if not self.min_price:
                self.min_price = 0
            if not self.max_price:
                self.max_price = 99999
            to_build.append(
                "price={}-{}".format(self.min_price, self.max_price))
        if self.page:
            to_build.append("page={}".format(self.page))
        if self.page_size:
            to_build.append("pageSize={}".format(self.page_size))
        if self.categories:
            for cat in self.categories:
                to_build.append("category={}".format(cat))

        if len(to_build) > 0:
            arguments = "&".join(to_build)
            url = self.BASE_URL + arguments
        return url
