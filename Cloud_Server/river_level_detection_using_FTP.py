import pickle
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

import base64

import ftplib

import os
import time
import datetime
import pytz

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

import numpy as np
import cv2

import torch
from torchvision import models, transforms

import shutil

from astral import LocationInfo
from astral.sun import sun


def load_trained_model():
    model = models.segmentation.deeplabv3_resnet50(pretrained=True)
    model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))

    model.cpu()
    
    checkpoint = torch.load("./trained_model/trained_model.pth", map_location=torch.device('cpu'))
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    model.eval()

    return model

def run_dnn_model(river_name):
    model = load_trained_model()   
    # Test img is the latest_img
    if river_name == "A":
        test_img = cv2.imread("./static/river_imgs/latest_river_A_img.jpg")
        reference_line = [(829, 173), (813, 260)]
        pixel_distance_empty = 88.45903006477066
        h_max = 2.7
        h_empty = 0.45
        river_max_level = 2.40
    else:
        test_img = cv2.imread("./static/river_imgs/latest_river_B_img.jpg")
        reference_line = [(603, 266), (610, 431)]
        pixel_distance_empty = 165.14841809717706
        h_max = 2.7
        h_empty = 0.25
        river_max_level = 1.50
    
    if test_img is not None:
        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        
        normalize = transforms.Normalize(mean=[0.5291628, 0.5135074, 0.45702627], std=[0.19916107, 0.18686345, 0.1919754])

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=224),
            transforms.CenterCrop(size=(224, 224)),
            transforms.ToTensor(),
            normalize])

        test_img_tensor = transform(test_img)
        test_img_tensor = test_img_tensor[None, :, :]

        output = model(test_img_tensor)["out"]
        _, preds = torch.max(output, 1)
        preds = preds[0, :, :].detach().cpu().numpy()

        obj_ids = np.unique(preds)
        masks = preds == obj_ids[:, None, None]

        segmentation_rgb = np.zeros(shape=(masks[0].shape[0], masks[0].shape[1], 3), dtype=np.uint8)
        
        try:
            segmentation_rgb[:, :, 0] = masks[1] * 255
            
            river_level_img = segmentation_rgb[:, :, 0]
            
            river_level_img = cv2.resize(river_level_img, (720,720), cv2.INTER_NEAREST)
                        
            segmentation_img = np.zeros(shape=(river_level_img.shape[0], river_level_img.shape[1], 3), dtype=np.uint8)
            segmentation_img[:, :, 0] = river_level_img
            
            lines_img = np.zeros(shape=test_img.shape, dtype=np.uint8)
            seg_img = np.zeros(shape=test_img.shape, dtype=np.uint8)
            
            # compute center offset
            x = (test_img.shape[1] - 720) // 2
            y = (test_img.shape[0] - 720) // 2
            
            seg_img[y: (y + 720), x: (x + 720), :] = segmentation_img
            
            x_0 = reference_line[0][0]
            y_0 = reference_line[0][1]
            
            x_1 = reference_line[1][0]
            y_1 = reference_line[1][1]
            
            slope_ref_line = (y_1 - y_0)/(x_1 - x_0)
            
            if slope_ref_line > 0:
                start_idx = 0
                end_idx = test_img.shape[1]
                step = 0.2
            else:
                start_idx = test_img.shape[1]
                end_idx = 0
                step = -0.2
                
            for x_intersect in np.arange(start_idx, end_idx, step):
                y_intersect = int( slope_ref_line*(x_intersect - x_0) + y_0)
                
                if y_intersect < seg_img.shape[0] and y_intersect > 0:
                    if seg_img[y_intersect, int(x_intersect), 0] == 255:
                        break
                    
            x_intersect = int(x_intersect)
            test_img = cv2.addWeighted(test_img, 0.8, seg_img, 0.2, 0)
            
            distance_in_pixel = np.sqrt((x_0 - x_intersect)**2 + (y_0 - y_intersect)**2)
            
            # Compute river level
            river_level_in_meters = np.around(h_max - ((h_max - h_empty)*distance_in_pixel/pixel_distance_empty), 4)
            
            if y_intersect > y_0 and river_level_in_meters < river_max_level:    
                img_text = "River level: {:.4f} m".format(river_level_in_meters)
                
                cv2.putText(test_img, img_text, (x + 15, y + 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
            else:
                river_level_in_meters = river_max_level
                img_text_0 = "River level is above " + str(river_max_level) + " m!"
                img_text_1 = "Flood Alert! Take Shelter Now!"

                cv2.putText(test_img, img_text_0, (x + 15, y + 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 0), 2)

                cv2.putText(test_img, img_text_1, (x+ 15, y + 145), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 0), 2)

            test_img = cv2.line(test_img, (x_0, y_0), (x_intersect, y_intersect), color=(0, 255, 0), thickness=2)
            test_img = cv2.circle(test_img, (x_0, y_0), 3, color=(0, 0, 255), thickness=-1)
            test_img = cv2.circle(test_img, (x_intersect, y_intersect), 3, color=(255, 0, 0), thickness=-1)
                    
            test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)

            if river_name == "A":
                cv2.imwrite("./static/river_imgs/river_A_level.jpg", test_img[y: (y + 720), x: (x + 720), :])
            else:
                cv2.imwrite("./static/river_imgs/river_B_level.jpg", test_img[y: (y + 720), x: (x + 720), :])
                
        except:
            print("Error detecting the water surface...")
            river_level_in_meters = 0.0
    else:
        river_level_in_meters = 0.0

    return river_level_in_meters
    
def verify_river_levels(night_time):
    river_A_level_in_meters = run_dnn_model("A")
    river_B_level_in_meters = run_dnn_model("B")

    with open("./river_level_variables.pickle", "wb") as f:
        pickle.dump([river_A_level_in_meters, river_B_level_in_meters, night_time], f)
    
    return river_A_level_in_meters, river_B_level_in_meters
    
def send_images_to_server(FTP_host, FTP_port, user_root, pwd_root):
    FTP = ftplib.FTP()
    
    retry = True
    while (retry):
        try:
            FTP.connect(FTP_host, FTP_port)
            FTP.login(user_root, pwd_root)
            retry = False
            
            FTP.cwd("/flash_flooding_alert_system/")
            with open("river_level_variables.pickle", "rb") as f:
                FTP.storbinary("STOR river_level_variables.pickle", f)

            FTP.cwd("/flash_flooding_alert_system/static/river_imgs/")
            with open("./static/river_imgs/river_A_level.jpg", "rb") as f:
                FTP.storbinary("STOR river_A_level.jpg", f)

            with open("./static/river_imgs/river_B_level.jpg", "rb") as f:
                FTP.storbinary("STOR river_B_level.jpg", f)
            
            FTP.quit()
        except:
            print("Error connecting to server... Retrying...")
            retry = True


def river_level_detection_loop():
    print("River level detection started")

    max_river_A = 2.40
    max_river_B = 1.50
    
    timezone = 'Brazil/East'
    now = datetime.datetime.now(pytz.timezone(timezone))
    city = LocationInfo("São Carlos", region="Brazil", timezone=timezone, latitude=-22, longitude=-47.890833)
    s = sun(city.observer, date=now, tzinfo=city.timezone)
    
    init_run_time = s['sunrise']
    end_run_time = s['sunset']

    FTP_host = 'FTP IP'
    FTP_port = PORT
    user_root = 'user_root'
    pwd_root = 'some_password'

    ################################################################################################
    night_time = 0
    alert_sent = False
    
    river_A_level_in_meters, river_B_level_in_meters = verify_river_levels(night_time)
    
    send_images_to_server(FTP_host, FTP_port, user_root, pwd_root)

    if (river_A_level_in_meters >= max_river_A or river_B_level_in_meters >= max_river_B) and not alert_sent:
        send_alert_email()
        alert_sent = True
                
    if river_A_level_in_meters < max_river_A and river_B_level_in_meters < max_river_B:
        alert_sent = False

    ################################################################################################
    FTP = ftplib.FTP()
    retry = True
    while (retry):
        try:
            FTP.connect(FTP_host, FTP_port)
            # SOME_USER is where our images are stored in our cloud server
            FTP.login('SOME_USER', 'some_password')
            retry = False
            
            FTP.cwd("/" + str(now.year) + "/")

            print("Retrieving new images...")

            list_imgs = FTP.nlst("-t")

            list_imgs_shop = [i for i in list_imgs if i.split(".")[0].split("-")[-1] == "SHOP2"]

            last_img_shop = list_imgs_shop[0]

            list_imgs_sesc = [i for i in list_imgs if i.split(".")[0].split("-")[-1] == "SESC2"]

            last_img_sesc = list_imgs_sesc[0]
        except:
            print("Error connecting to server... Retrying...")
            retry = True   
    
    if now >= init_run_time and now < end_run_time:        
        with open("./static/river_imgs/latest_river_A_img.jpg", "wb") as f:
            try:
                FTP.retrbinary("RETR " + last_img_shop, f.write)
            except:
                print("Error while trying to connect to server...")
        
        with open("./static/river_imgs/latest_river_B_img.jpg", "wb") as f:
            try:
                FTP.retrbinary("RETR " + last_img_sesc, f.write)
            except:
                print("Error while trying to connect to server...")
        
        try:
            FTP.quit()
        except:
            print("Error while trying to connect to server...")
        
        night_time = 0
        
        river_A_level_in_meters, river_B_level_in_meters = verify_river_levels(night_time)
            
        send_images_to_server(FTP_host, FTP_port, user_root, pwd_root)

        if (river_A_level_in_meters >= max_river_A or river_B_level_in_meters >= max_river_B) and not alert_sent:
            send_alert_email()
            alert_sent = True
                
        if river_A_level_in_meters < max_river_A and river_B_level_in_meters < max_river_B:
            alert_sent = False
    else:
        night_time = 1
        
        river_A_level_in_meters, river_B_level_in_meters = verify_river_levels(night_time)
            
        send_images_to_server(FTP_host, FTP_port, user_root, pwd_root)

    # Keep updating the images
    while True:
        # Get current time, sunrise time, and sunset time #
        now = datetime.datetime.now(pytz.timezone(timezone))
        city = LocationInfo("São Carlos", region="Brazil", timezone=timezone, latitude=-22, longitude=-47.890833)
        s = sun(city.observer, date=now, tzinfo=city.timezone)
        
        init_run_time = s['sunrise']
        end_run_time = s['sunset']
        ###################################################

        print("Waiting for new images...")
        FTP = ftplib.FTP()
        retry = True
        while (retry):
            try:
                FTP.connect(FTP_host, FTP_port)
                # SOME_USER is where our images are stored in our cloud server
                FTP.login('SOME_USER', 'SOME_PASSWORD')
                retry = False
                
                FTP.cwd("/" + str(now.year) + "/")
                list_imgs = FTP.nlst("-t")

                list_imgs_shop = [i for i in list_imgs if i.split(".")[0].split("-")[-1] == "SHOP2"]
                last_img_shop_in_ftp_server = list_imgs_shop[0]

                list_imgs_sesc = [i for i in list_imgs if i.split(".")[0].split("-")[-1] == "SESC2"]
                last_img_sesc_in_ftp_server = list_imgs_sesc[0]
            except:
                print("Error connecting to server... Retrying...")
                retry = True
        
        if (last_img_shop_in_ftp_server != last_img_shop) or (last_img_sesc_in_ftp_server != last_img_sesc):
            print("Newest image found in the FTP server!")
            
            # Move last img in the FTP server to the the last_img
            last_img_shop = last_img_shop_in_ftp_server
            last_img_sesc = last_img_sesc_in_ftp_server

            with open("./static/river_imgs/latest_river_A_img.jpg", "wb") as f:
                try:
                    FTP.retrbinary("RETR " + last_img_shop, f.write)
                except:
                    print("Error connecting to server...")

            with open("./static/river_imgs/latest_river_B_img.jpg", "wb") as f:
                try:
                    FTP.retrbinary("RETR " + last_img_sesc, f.write)
                except:
                    print("Error connecting to server...")

            try:
                FTP.quit()
            except:
                print("Error connecting to server...")
            
            if now >= init_run_time and now < end_run_time:
                night_time = 0
                
                river_A_level_in_meters, river_B_level_in_meters = verify_river_levels(night_time)
            
                send_images_to_server(FTP_host, FTP_port, user_root, pwd_root)

                if (river_A_level_in_meters >= max_river_A or river_B_level_in_meters >= max_river_B) and not alert_sent:
                    send_alert_email()
                    alert_sent = True
                
                if river_A_level_in_meters < max_river_A and river_B_level_in_meters < max_river_B:
                    alert_sent = False
            else:
                night_time = 1

                river_A_level_in_meters, river_B_level_in_meters = verify_river_levels(night_time)
            
                send_images_to_server(FTP_host, FTP_port, user_root, pwd_root)

                alert_sent = False
        
        # Wait X minutes to update
        time.sleep(120)

def send_alert_email():    
    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('gmail', 'v1', credentials=creds)

    receiver_email_list = ["someone@email.com"]

    for i in range(len(receiver_email_list)):
        filenameA = "./static/river_imgs/river_A_level.jpg"
        fpA = open(filenameA, 'rb')

        filenameB = "./static/river_imgs/river_B_level.jpg"
        fpB = open(filenameB, 'rb')
        
        message = MIMEMultipart()
        message['from'] = "email from where the system sends alerts @ email.com"
        message['subject'] = "River Flooding Alert in São Carlos, SP!"

        msg = MIMEText("There is a River Flooding Alert in São Carlos, SP!\nTake shelter now!\n\nThe current river levels are attached as images to this email.\n\nE-Noe Flooding Alert System.")
        message.attach(msg)
        
        msg = MIMEImage(fpA.read(), _subtype=1)
        msg.add_header('Content-Disposition', 'attachment', filename="river_A_level.jpg")
        message.attach(msg)

        msg = MIMEImage(fpB.read(), _subtype=1)
        msg.add_header('Content-Disposition', 'attachment', filename="river_B_level.jpg")
        message.attach(msg)

        message["To"] = receiver_email_list[i]
        message["Bcc"] = receiver_email_list[i]  # Recommended for mass emails

        raw_message = {'raw': base64.urlsafe_b64encode(message.as_string().encode("utf-8")).decode("utf-8")}

        message = service.users().messages().send(userId='me', body=raw_message).execute()

        fpA.close()
        fpB.close()
        
        message = None

if __name__ == "__main__":
    river_level_detection_loop()
