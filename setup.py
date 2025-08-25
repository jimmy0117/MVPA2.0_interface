import requests

def download_from_dropbox(shared_url, dest_path):
    # 轉換成直接下載連結
    download_url = shared_url.replace('www.dropbox.com', 'dl.dropboxusercontent.com')
    download_url = download_url.replace('?dl=0', '').replace('?dl=1', '')
    r = requests.get(download_url, stream=True)
    r.raise_for_status()
    with open(dest_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

def __main__():
    url_list = [
        'https://www.dropbox.com/scl/fi/wwvjk53jqrpr6qw3wy2zm/ffmpeg.exe?rlkey=o71zs6bnk5zbbygpzyakmyonp&st=bi2gojhl&dl=0',
        'https://www.dropbox.com/scl/fi/ef6jwfqr3qm4pwe9q4nul/ffplay.exe?rlkey=tjclrpkht6yo6qmmmnf7023ku&st=y3d26r81&dl=0',
        'https://www.dropbox.com/scl/fi/r416xd5bv3dar8nmgj7m4/ffprobe.exe?rlkey=vbcxg80zatmi0jxgwjitwjll4&st=sqf3zonh&dl=0',
        {'Models': [
            'https://www.dropbox.com/scl/fi/jx1c2y11r2amh76wv0684/function_pred_SVM.py?rlkey=f2nw9qac3q0mtqce07kpwh98q&st=vt3jz20y&dl=0',
			'https://www.dropbox.com/scl/fi/1use7ny76k8ymily981mb/best_svm_model.pkl?rlkey=hy9sh8vr9y0ab7gvr07xb6rg8&st=rdxjsy1s&dl=0',
            'https://www.dropbox.com/scl/fi/orrcyt3go73b9itaq7grl/function_pred.py?rlkey=08crbeft8frr577aefjk6o9xm&st=q3xhrwlg&dl=0',
            'https://www.dropbox.com/scl/fi/p5berqdxlz0jnj623nnak/smote_cnn_model_0.86.h5?rlkey=8osxu647es9ekp6sfa791frs2&st=qk3qr45z&dl=0',
            'https://www.dropbox.com/scl/fi/9rjmsq7m26yxw9bdsal8s/function_pred.py?rlkey=o8ye41ptwqjprszztv2cnut1y&st=gp5a9kiv&dl=0',
            'https://www.dropbox.com/scl/fi/mmzecfr378l0czwfkpkty/smote_non_cnn_model_0.84.h5?rlkey=6bhoqqdgpj532h5g8ifd1uv6z&st=srt3mpiu&dl=0',
        ]}
        # 這裡可以放多個 Dropbox 連結
    ]
    for url in url_list:
        if isinstance(url, str):
            print(f"Downloading from {url}...")
            dest_path = 'd:/MVPA2.0_interface/' + url.split('/')[-1].split('?')[0]
            download_from_dropbox(url, dest_path)
            print(f"Saved to {dest_path}")
        elif isinstance(url, dict):
            for key in url:
                i = 1
                for link in url[key]:
                    print(f"Downloading from {link}...")
                    dest_path = 'd:/MVPA2.0_interface/' + f'{key}/{i}_' + link.split('/')[-1].split('?')[0]
                    download_from_dropbox(link, dest_path)
                    print(f"Saved to {dest_path}")
                    i+=1

if __name__ == "__main__":
    __main__()