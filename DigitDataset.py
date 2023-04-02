from torchvision import datasets, transforms
from PIL import Image, ImageDraw, ImageFont
import torch
import glob
import time
import random

class DigitDataset(torch.utils.data.Dataset):
    def __init__(self, image_size, number):
        self.fonts= glob.glob("fonts/*.ttf")
        self.fonts_dict={}
        self.number=number
        self.digits_=[None]*self.__len__()
        self.image_size=image_size
        self.generate_all()
    
    def __len__(self):
        return self.number
    
    def __getitem__(self,index):
        return self.digits_[index]
    
    def generate_all(self):
        print("Generating the digit dataset")
        t_start=time.monotonic()
        for p in range(self.__len__()):
            if p%10000==0:
                print(f"{p} of {self.__len__()}...")
            self.digits_[p]=self.generate_digits()
        print(f"Done, dT={time.monotonic()-t_start}")
    
    def generate_digits(self):
        digit=random.randint(0,9)
        data=self.generate_digit_pil(digit)
        return data,digit

    def generate_digit_pil(self, digit:int):
        text=str(digit)
        area_size=2*self.image_size
        img=Image.new("L",(area_size,area_size),(0,))
        draw= ImageDraw.Draw(img)
        font_name,font_size=random.choice(self.fonts),random.randint(48,64)
        font_key=f"{font_name}-{font_size}"
        if font_key not in self.fonts_dict:
            self.fonts_dict[font_key]=ImageFont.truetype(font_name, font_size)
        
        font=self.fonts_dict[font_key]
        text_x= area_size//2+random.randint(-2,2)
        text_y=area_size//2+random.randint(-1,1)
        draw.text((text_x,text_y),text,(255,),font=font,anchor="mm")
        transform= transforms.Compose([transforms.Resize([self.image_size,
                                                          self.image_size]),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,),(0.3081,))])
        resized= transform(img)[0].unsqueeze(0)
        return resized


