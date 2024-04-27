import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver import ChromeOptions
import time


# Scrapping from Amazon

def parse_image_urls(soup, classes, location, source):
    for a in soup.findAll(attrs={"class": classes}):
        name = a.find(location)
        if name not in results:
            results.append(name.get(source))

def get_img_links(homepage, recurrent_links, no_of_pages, class_name, location, source):
    options = ChromeOptions()
    options.add_argument("--headless=new")
    driver = webdriver.Chrome(options=options)
    url = homepage
    driver.get(url)
    content = driver.page_source
    soup = BeautifulSoup(content, "html.parser")
    parse_image_urls(soup, class_name, location, source)
    print(f"Page 1 done")
    time.sleep(2)
    # loop through the pages
    for i in range(2, no_of_pages + 1):
        options = ChromeOptions()
        options.add_argument("--headless=new")
        driver = webdriver.Chrome(options=options)
        url = recurrent_links + str(i)
        driver.get(url)
        content = driver.page_source
        soup = BeautifulSoup(content, "html.parser")
        parse_image_urls(soup, class_name, location, source)
        print(f"Finished scraping page {url}")
        print(len(results))
        time.sleep(2)
    driver.quit()
    return results