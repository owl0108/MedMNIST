MEDMNIST_DESCRIPTION = {
    'pathmnist' : """The PathMNIST is based on a prior study for predicting survival from colorectal cancer histology slides, providing a dataset
                    (NCT-CRC-HE-100K) of 100,000 non-overlapping image patches from hematoxylin & eosin stained histological images, and
                    a test dataset (CRC-VAL-HE-7K) of 7,180 image patches from a different clinical center. The dataset is comprised of 9 types
                    of tissues, resulting in a multi-class classification task.""",
    'octmnist': """The OCTMNIST is based on a prior dataset of 109,309 valid optical coherence tomography (OCT) images for retinal
                    diseases. The dataset is comprised of 4 diagnosis categories, leading to a multi-class classification task.""",
    'pneumoniamnist':"""The PneumoniaMNIST is based on a prior dataset of 5,856 pediatric chest X-Ray images. The task is binary-class
                    classification of pneumonia against normal.""",
    'chestmnist': """The ChestMNIST is based on the NIH-ChestXray14 dataset, a dataset comprising 112,120 frontal-view X-Ray images
                    of 30,805 unique patients with the text-mined 14 disease labels, which could be formulized as a multi-label binary-class
                    classification task.""",
    'dermamnist': """The DermaMNIST is based on the HAM10000, a large collection of multi-source dermatoscopic images of common
                    pigmented skin lesions. The dataset consists of 10,015 dermatoscopic images categorized as 7 different diseases, formulized as
                    a multi-class classification task.""",
    'retinamnist':"""The RetinaMNIST is based on the DeepDRiD challenge, which provides a dataset of 1,600 retina fundus images. The task
                    is ordinal regression for 5-level grading of diabetic retinopathy severity.""",
    'breastmnist':"""The BreastMNIST is based on a dataset of 780 breast ultrasound images. It is categorized into 3 classes: normal, benign, and
                    malignant. As we use low-resolution images, we simplify the task into binary classification by combining normal and benign as
                    positive and classifying them against malignant as negative.""",
    'bloodmnist':"""The BloodMNIST is based on a dataset of individual normal cells, captured from individuals without infection, hematologic
                    or oncologic disease and free of any pharmacologic treatment at the moment of blood collection. It contains a total of 17,092
                    images and is organized into 8 classes.""",
    'tissuemnist':"""We use the BBBC051, available from the Broad Bioimage Benchmark Collection. The dataset contains 236,386 human
                    kidney cortex cells, segmented from 3 reference tissue specimens and organized into 8 categories.""",
    'organamnist':"""The OrganAMNIST is based on 3D computed tomography (CT) images from Liver Tumor Segmentation Benchmark
                    (LiTS). It is renamed from OrganMNIST_Axial (in MedMNIST v1
                    ) for simplicity. We use boundingbox annotations of 11 body organs from another study to obtain the organ labels. Hounsfield-Unit (HU) of the 3D images are
                    transformed into gray-scale with an abdominal window. We crop 2D images from the center slices of the 3D bounding boxes in the axial view (plane).""",
    'organcmnist':"""The OrganCMNIST is based on 3D computed tomography (CT) images from Liver Tumor Segmentation Benchmark
                    (LiTS). It is renamed from OrganMNIST_Coronal (in MedMNIST v1
                    ) for simplicity. We use boundingbox annotations of 11 body organs from another study to obtain the organ labels. Hounsfield-Unit (HU) of the 3D images are
                    transformed into gray-scale with an abdominal window. We crop 2D images from the center slices of the 3D bounding boxes in the coronal view (plane).""",
    'organsmnist':"""The OrganSMNIST is based on 3D computed tomography (CT) images from Liver Tumor Segmentation Benchmark
                    (LiTS). It is renamed from OrganMNIST_Sagittal (in MedMNIST v1
                    ) for simplicity. We use boundingbox annotations of 11 body organs from another study to obtain the organ labels. Hounsfield-Unit (HU) of the 3D images are
                    transformed into gray-scale with an abdominal window. We crop 2D images from the center slices of the 3D bounding boxes in the sagittal view (plane)."""
}