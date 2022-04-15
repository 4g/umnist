## Setup

    git clone https://github.com/4g/umnist
    git checkout master
    cd umnist
    git submodule update --init
    pip install -r requirements.txt
    pip install -r yolov5/requirements.txt
    python pipeline.py --task make_dirs

This creates following directories:

    store
    ├── data
    │   ├── competition
    │   ├── generated
    │        └── yolo
    └── models
        ├── classifier
        └── yolo

Paste your test images folder inside store/data/competition/. 
So `umnist/store/data/competition/test/` will have competition jpeg images. 

#### Infer with pretrained model
Trained models are saved in the repo and can be used to regenerate the results. Following command generates a yolo_submission.csv file.

    python pipeline.py --task infer_pretrained


## Train from scratch:

#### Create directories
    python pipeline.py --task make_dirs

Copy paste the test images folder to store/data/competition/ as described above

#### Generate synthetic detector data

    python pipeline.py --task generate_yolo

#### Generate synthetic classifier data

    python pipeline.py --task generate_classifier

#### Train yolo

    python pipeline.py --task train_yolo

#### Train classifier

    python pipeline.py --task train_classifier

#### Infer with newly trained models:
    python pipeline.py --task infer
