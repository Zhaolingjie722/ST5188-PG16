we have two resources of generated data
11labs:

1. we have speech.txt as the text we want to convert
2. we have password.env with api key in it in form of
   ELEVENLABS_API_KEY=sk_aa78c271f2e2242bbe1b718f998bf164038326451b5a5dc0
   the api key can be generated in https://elevenlabs.io/app/settings/api-keys
3. we have reader_ids.txt with ids of 20 speakers, if users want to add or change speakers, they can go search website
   of elevenlabs, the file have format that one line one id
4. #run
   python3 11lab.py

diffgan

1. git clone https://github.com/keonlee9420/DiffGAN-TTS.git
2. note all the packages are based on python 3.7,make sure run the code in environment with python 3.7
3. #install needed packages
   pip3 install -r requirements.txt
4. #download the pretrained model:
   gdown LSzPHzq5NFlitxsVmaJLyt5vDvSjvn8N -O /DiffGan-TTS/output/ckpt/DATASET_naive/200000.pth.tar
5. make sure text.txt contain the text you want to convert
6. run
   python3 diffgan.py
7. run
   bash commands.sh
