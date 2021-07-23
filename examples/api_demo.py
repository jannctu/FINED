import requests

def ping(url):
    response = requests.get(url)
    d = response.json()
    print(d['Data'])

def test_image(url,image_file):
    files = {
        'file' : open(image_file, 'rb')
    }
    response = requests.request("POST", url, files=files)
    with open("edge.png", 'wb') as f:
        f.write(response.content)

if __name__ == "__main__":
    ping("http://lab1.jankristanto.com:8000/")
    url = 'http://lab1.jankristanto.com:8000/predict/image'
    fname = 'pandas.jpg'
    test_image(url,fname)