import splitfolders

#인풋 폴더
"""
input/
    class1/
        img1.jpg
        img2.jpg
        ...
    class2/
        imgWhatever.jpg
        ...
    ...
"""
input_folder = '/home/tlab/dataset/image'
#아웃풋 폴더
"""
output/
    train/
        class1/
            img1.jpg
            ...
        class2/
            imga.jpg
            ...
    val/
        class1/
            img2.jpg
            ...
        class2/
            imgb.jpg
            ...
    test/
        class1/
            img3.jpg
            ...
        class2/
            imgc.jpg
            ...           
"""
out_foloder = '/home/tlab/dataset/output'
#학습비율
train_ratio = 0.6
#검증비율
val_ratio = 0.2
#테스트비율
test_ratio = 0.2
#라벨링 폴더와 같이있으면 2 없으면 None
anno_set = 2
#파일을 복사하지않고 움빅일경우 True
move_opt = False

splitfolders.ratio(input_folder,output=out_foloder, seed =833, ratio=(train_ratio,val_ratio,test_ratio), group_prefix=anno_set, move=False)
