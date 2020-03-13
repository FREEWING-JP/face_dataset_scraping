import requests
from bs4 import BeautifulSoup
import urllib.request
import argparse
import os


def main(url,output_dir):
    #結果を収集するディレクトリ生成
    os.makedirs(output_dir, exist_ok=True)

    #soupがurlの指すページを見る
    res = requests.get(url)
    soup = BeautifulSoup(res.text, 'html.parser')

    #soup(webページ)上のアンカ要素をリストとして取得
    links = [url.get('href') for url in soup.find_all('a')]
    links = list(dict.fromkeys(links))

    #アンカ要素のうち、"archives"を含むリンクについてsub_funcを実行
    count = 0
    progress_len = len(links)
    progress_count = 0
    for link in links:
        print(str(progress_count)+"/"+str(progress_len))
        progress_count += 1
        if type(link) != type("string"):
            continue
        if "archives" in link:
            sub_func(link, count, output_dir)
            count += 1


def sub_func(link, count, output_dir):
    #結果を収集するサブディレクトリ生成
    os.makedirs(output_dir+"/"+str(count), exist_ok=True)

    #再びsoupがサブurlの指すページを見る
    res = requests.get(link)
    soup = BeautifulSoup(res.text, 'html.parser')

    #今度はアンカがpngやjpgだったときのみ動作
    links = [url.get('href') for url in soup.find_all('a')]
    index = 0
    for link in links:
        if type(link) != type("string"):
            continue

        #ダウンロード時にエラーが起こることもあるのでそこで止まらないようにする
        try: 
            if (".png" in link):
                urllib.request.urlretrieve(link,"{0}/{1}/{2}.png".format(output_dir,count,index))
                index += 1
            elif (".jpg" in link):
                urllib.request.urlretrieve(link,"{0}/{1}/{2}.jpg".format(output_dir,count,index))
                index += 1
        except:
            print('Download Error')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='scraping images program')
    parser.add_argument("url",help="the web page url to do scraping images")
    parser.add_argument("-o","--output",help="the output directory", default="./scraped")
    args = parser.parse_args()    
    main(args.url, args.output)