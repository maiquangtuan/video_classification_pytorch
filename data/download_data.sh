pip install av
wget https://raw.githubusercontent.com/pytorch/vision/6de158c473b83cf43344a0651d7c01128c7850e6/references/video_classification/transforms.py
# Download HMDB51 data and splits from serre lab website
wget http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar
wget http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar
mkdir -p video_data test_train_splits
unrar e test_train_splits.rar test_train_splits
rm test_train_splits.rar
unrar e hmdb51_org.rar 
rm hmdb51_org.rar
mv *.rar video_data
python org_data.py
rm video_data/*.rar
