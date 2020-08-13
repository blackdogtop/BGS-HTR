import requests,sys
from bs4 import BeautifulSoup
import os


class Downloader:
    """
    类说明: download scanned borehole records images from BGS website
    :param
    :return
    Modify:
        18.05.2020
    """

    def __init__(self):
        self.img_path = '../data/downloaded-BGS-records'  # where the images to store
        self.server_page = 'http://scans.bgs.ac.uk/sobi_scans/internal/boreholes/pages/'
        self.grids = {}  # store gird names and index
        # if not exist then make dir
        if not os.path.exists(self.img_path):
            os.mkdir(self.img_path)

    def download_img(self, url, filename):
        """
        函数说明: download an image to the specific path
        :param url:the url of the image
        :param filename:filename of the img(contain path)
        :return none
        Modify:
            18.05.2020
        """
        if not os.path.exists(self.img_path):
            os.mkdir(self.img_path)
        req = requests.get(url, stream=True)
        with open(filename, 'wb') as f:
            for image in req.iter_content(chunk_size=1024):
                if image:
                    f.write(image)
                    f.flush

    def get_img_by_id(self, targets):
        """
        函数说明: get all images of a BGS ID/Reference (a BGS ID/Reference probably has several imgs)
        :param
            target:scanned records of a BGS ID/Reference
        :return none
        Modify:
            08.06.2020
        """
        req = requests.get(url=targets)
        html = req.text
        div_bf = BeautifulSoup(html)
        div = div_bf.find_all('select', id='page')
        option_bf = BeautifulSoup(str(div))
        option = option_bf.find_all('option')
        for value in option:
            img_target = value.get('value')
            img_target = img_target + '.html'  # 添加后缀以免直接解析
            req_id = requests.get(url=img_target)
            html_id = req_id.text
            # print(html_id)
            img_bf = BeautifulSoup(html_id)
            img = img_bf.find_all('img', id='image')
            try:
                img_src = [i.get('src') for i in img][0]
                # print(img_src)
            except IndexError:  # Image Unavailable
                continue
            else:
                img_id = img_src.split('/')[-3] + '_'
                img_name = img_src.split('/')[-1]
                filename = '{}{}{}'.format(self.img_path + '/', img_id, img_name)
                self.download_img(img_src, filename)

    def get_img_by_page(self, page):
        """
        函数说明: get all images in a page
        :param
            page: which page to download
        :return
        Modify:
            08.06.2020 - fix bug
        """
        url = self.server_page + str(page)
        html = requests.get(url, timeout=(3, 20)).text  # connect timeout, read timeout
        li_bf = BeautifulSoup(html)
        li = li_bf.find_all('li')
        a_bf = BeautifulSoup(str(li))
        a = a_bf.find_all('a')
        print(('Downloading The {} Page Images............').format(page))
        for i, each in enumerate(a):
            print(('Downloading the {} Record In The Page').format(i))
            target_url = each.get('href')
            target_url = target_url + '.html'  # 添加后缀以免直接解析
            # print(target_url)
            self.get_img_by_id(target_url)

    def get_grid_name(self, page_num=int):
        """
        函数说明: get the first grid name in a page
        :param
            page_num: the page number
        :return
            grid_name: the first grid name in the page
        Modify:
            19.05.2020
        """
        page_url = self.server_page + str(page_num)
        html = requests.get(url=page_url).text
        li_bf = BeautifulSoup(html)
        li = li_bf.find_all('li')
        grid_name = li[0].string[:2]

        return grid_name

    def get_grid_index(self, page_num=int):
        """
        函数说明: get a district grid index
        :param
            page: the started page number
        :return
            index: the range(start page and end page) of a grid
        Modify:
            19.05.2020
        """
        init_page = low_page = page_num
        val = self.get_grid_name(low_page)

        high_page = low_page + 100
        grid_name = self.get_grid_name(high_page)

        while val == grid_name:  # 确定high_page的值确保high_page的li的grid name与val不同
            high_page += (high_page - low_page)
            grid_name = self.get_grid_name(high_page)

        mid_page = (low_page + high_page) // 2
        mid_grid_name = self.get_grid_name(mid_page)
        next_page = mid_page + 1
        next_grid_name = self.get_grid_name(next_page)

        while low_page <= high_page and not (mid_grid_name == val and next_grid_name != val):
            mid_page = (low_page + high_page) // 2
            mid_grid_name = self.get_grid_name(mid_page)
            next_page = mid_page + 1
            next_grid_name = self.get_grid_name(next_page)
            if next_grid_name == val:
                low_page = mid_page + 1
            elif mid_grid_name != val:
                high_page = mid_page - 1

        index = [init_page, mid_page]
        return index

    def get_bgs_grids_index(self):
        """
        函数说明: get all UK BGS grids {name : index}
        :param
        :return
        Midify:
            19.05.2020
        """
        started_page = 1
        while started_page <= 18561:  # not get north part of UK scanned records
            name = self.get_grid_name(started_page)
            index = self.get_grid_index(started_page)
            self.grids[name] = index
            started_page = index[-1] + 1
            print("进度为{:.2f} %".format((started_page / 18561) * 100))

    def get_img_by_grid(self, grid=str):
        """
        函数说明: get all images of a grid
        :param
            grid: the name of grid, such as "SD", "SE"
        :return
        Modify:
            19.05.2020
        """
        print('Getting Borehole Records Index......')
        self.get_bgs_grids_index()
        # print(self.grids)
        try:
            grid_index = self.grids[grid]
        except KeyError:
            print('Error')
        else:
            for page in range(grid_index[0], grid_index[1] + 1):
                print("进度为{:.2f} %".format(((page - grid_index[0]) / (grid_index[-1] - grid_index[0])) * 100))
                self.get_img_by_page(page)

if __name__ == '__main__':
    downloader = Downloader()

    # download according to BNG square
    downloader.get_img_by_grid('SD')

    # download according to BGS index page
    # for i in range(2):
    #     downloader.get_img_by_page(i)

    # download according to BGS ID - the url refers to BGS ID: 3
    # downloader.get_img_by_id('http://scans.bgs.ac.uk/sobi_scans/boreholes/3/images/10000007.html')