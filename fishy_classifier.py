#import fastai library

from fastai import *
from fastai.vision import *

#import google colab files
from google.colab import files
uploaded = files.upload()

#choose the file(usually zip) and unzip it 
!unzip train.zip

#classes you need to classify
classes = ['ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT']

#provide the path and verify them
path = Path('train/')
for c in classes:
  print(c)
  verify_images(path/c, delete=True, max_workers=8)
  
#Create Image Databunch 
np.random.seed(42)
data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,
        ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)

#display images
data.classes
data.show_batch(rows=3, figsize=(7,8))
data.classes, data.c, len(data.train_ds), len(data.valid_ds)

#create model
learn = create_cnn(data, models.resnet34, metrics=error_rate)

learn.fit_one_cycle(8)

#save model
learn.save('stage-1')

#To tweak or change few hyperparameters we need to unfreeze it and make the changes.
#In this way the model can be saved and new model can be loaded accordingly.
learn.unfreeze()

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(2, max_lr=slice(lr1,lr2))       #replace lr1,lr2 with appropriate learning rates

learn.load('stage-2')

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()

#Testing 
img = open_image(path/'DOL'/'img.jpg')   % path/'class'/'image_file'
img 

#prediction
pred_class,pred_idx,outputs = learn.predict(img)
pred_class

#EOD


  
  
  
  
  
  
  
  
  
  
  
  
  
  
