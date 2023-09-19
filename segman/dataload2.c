class Segmnt_Dataset : public torch::data::datasets::Dataset<Segmnt_Dataset> {
private: ///fields
   //List of pairs <Image_path, Mask_path>. Mask_path is empty for 'good' (no-defect) images (photos).
   double split_share_ = 0.0; //probability data goes to data2_ for other dataset
   Data_type data_ = {};  //data storage for this dataset
   Data_type data2_ = {}; //data storage for other, split dataset
   std::string dts_dir_ = ""; //where image class subdirs are. (like 'c:/datasets/HTCC_Crack/Test set')
   int out_tnz_size_ = 0; //output height and width for all image and mask tensors. If 0 => smallest size of 1st image.
   int n_channels_ = -1; //(auto) output N channels in photo images =3 for RGB or =1 for b/w images.
   std::map<std::string, int> NG_class_names_ ; //list of 'Not-Good' image classes (has 'masks' sub-dir) with weights.
   std::map<std::string, int> NG_class_nums_;   //list of number samples in 'Not-Good' image classes.
   std::map<std::string, int> OK_class_names_ ; //list of 'Good' image classes (hasn't 'masks' sub-dir) with weights.
   std::map<std::string, int> OK_class_nums_;   //list of number samples in 'Good' image classes.
   ColorMap_T colormap_ ; //List of [BGR] colors in all segmentation masks for 'Not-Good' images.
   long n_bboxes_ = 8; //max number of bboxes per image for detection (for 120x120 img, 8 max area bboxes is ok, I guess).
   bool print_info_ = true; //?print dataset params after setup? (recomended)

private:
   //prepare data_ list of Images and Masks file pathes pairs from dts_dir_.
   void readInfo();

public:  ///methods

   //Construct dataset from root_dir (with images sub-dirs); 'train', 'test' or 'validation' (or 'val') keyword; internal split-list.
   Segmnt_Dataset(
      std::string dts_dir,   //like 'c:/datasets/HTCC_Crack/Test Set'
      double split_share = 0.0, //random part of data from dataset_dir for other dataset split (from [0. - 1.] float range).
      int out_tnz_chnl = 3,  //image tensor number of channels: 1 or 3 only.
      int out_tnz_size = 0,  //out (quadrate) tensors size. Input Images and Masks might be any size. If 0 => smallest size of 1st image.
      bool shuffle = true,   //initial random shuffle (or not) the order of images and masks (labels) in data_ list.
      bool print_info = true //?print dataset params after setup? (recomended)
   );
   Segmnt_Dataset() {}; //empty constructor. Prepare copy from other dataset split.
   
   void copy_split(Segmnt_Dataset src_dataset); //copy params and rest of the data (second split) from other segmentation dataset

   void add(Segmnt_Dataset src_dataset); //append data from other segmentation dataset

   //Counts escapes in DNN output labels (masks) batch, comparing to referense labels batch (sizes:[B,H,W])
   int count_escapes(at::Tensor ref_lbl, at::Tensor out_lbl);

   //Counts overkills in DNN output labels (masks) batch, comparing to referense labels batch (sizes:[B,H,W])
   int count_overkills(at::Tensor ref_lbl, at::Tensor out_lbl);


   torch::data::Example<> get(size_t index); //get <image_tensor, labels_tensor> pair. Standart function for all libtorch datasets.
   torch::optional<size_t> size() const; //get total number of input Images (and its Labels(Masks)) in this dataset. (Standart function.)
   std::pair<std::string, std::string> getImgPthMaskPth(size_t index); //get pair <Image_path, Mask_path> for this index.
   int64_t n_classes() const; //returns number of all Image classes (Good and Not-Good). 
   int64_t n_ok_classes() const; //returns number of 'Good' Image classes. 
   int64_t n_ng_classes() const; //returns number of 'Not-Good' Image classes (Good and Not-Good). 
   int64_t tnz_size() const; //returns output images (and masks) quadrate tensors width and height 
   int64_t n_img_channels() const; //returns photo images N of channels ( =3 for RGB, = 1 for b/w images)
   ColorMap_T colormap() const; //returns ColorMap (vec<[R,G,B]>) - list of colors used in Masks for Not-Good images.
   std::string Segmnt_Dataset::mode() const; //return "segmentation" if mode_=0; "classification" if mode_=1; else: exception 'not realised'.
   std::map<std::string, int> ok_class_names() const; //returns list of all 'Good' classnames.
   std::map<std::string, int> ng_class_names() const; //returns list of all 'Not-Good' classnames.
   
   Data_type get_split_data() {
      return data2_;
   };

   double split_share() const {
      return split_share_;
   };

   std::string root() const {
      return dts_dir_;
   };

   long len() const {
      return long(data_.size());
   };

   //get Image_path for this index.
   std::string getImgPath(size_t index) {
      auto img_mask_paths = getImgPthMaskPth(index);
      return img_mask_paths.first;
   }
   //get Mask path for this index.
   std::string getMaskPath(size_t index) {
      auto img_mask_paths = getImgPthMaskPth(index);
      return img_mask_paths.second;
   }

   //Saves data sample [image tensor; labels] to image file. Retuns "" if Ok or error string.
   bool savSegmSample(at::Tensor im, //images tensor (RGB;norm) size:[C,H,W] 
      at::Tensor lbl, //labels tensor; sizes:[H,W] for segm. mask or [Nbbs,5] for detection labeled bboxes {x1,y1,x2,y2,lbl}
      std::string sav_fl_path, //save file string path. Non-exist dirs will not be created (error)
      double alpha = 1., //transparency for masks or boxes added to image
      double scale = 1. //scale image and labels before save
   );

   //Transform data sample batch [image tensors; labels batch] to images grid and 
   //  saves it to given image file. Retuns "" if Ok or error string.
   std::string savSegmSampleBatch2Grid(at::Tensor im_bt, //images tensor (RGB;norm) size:[C,H,W] 
      at::Tensor lbl_bt, //labels tensors batch; sizes:[B,H,W] for segm. mask or [Nbbs,5] for detection labeled bboxes {x1,y1,x2,y2,lbl}
      std::string sav_fl_path, //save file string path. Non-exist dirs will not be created (error)
      double alpha = 1., //transparency for masks or boxes added to image
      double scale = 1., //scale image and labels before save
      int gap_sz = 2,  //grid gaps size (between scaled images in grid)
      uint8_t gap_val = 0  //grid gaps fill value. Same for all channels if C>1 (gray color).
   );

};