Setup
git clone https://github.com/4g/umnist

pip install -r requirements.txt

Infer with pretrained model:
Trained models are saved in the repo and can be used to regenerate the results. 


Train from scratch:


cd umnist
rm -r store
python pipeline.py make_dirs

Generate synthetic detector data:

Generate synthetic classifier data:

Train yolo:

Train classifier:

Infer with newly trained models:


make_dirs/generate_yolo/generate_classifier/train_yolo/train_classifier/set_pretrained/infer/infer_pretrained