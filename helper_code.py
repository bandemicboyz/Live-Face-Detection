# batch images to visualize n number of images at one time in this case its 4 images at once
image_generator = images.batch(4).as_numpy_iterator()
plot_images = image_generator.next()

# loop through and visualize each batch of images set shuffle to True when loading images into dataset for randomization
fig,ax = plt.subplots(ncols = 4,figsize = (20,20))
for idx,image in enumerate(plot_images):
    ax[idx].imshow(image)
plt.show()

#this code can be used to test the augmentation on a single image with its corresponding label
test = cv2.imread(os.path.join('data','train','images','9f7e9b02-ba0d-11ed-be93-201e88bb1731.jpg'))
print(test)
with open(os.path.join('data','train','labels','9f7e9b02-ba0d-11ed-be93-201e88bb1731.json'),'r') as f:
    label = json.load(f)
print(label)

# map coordinates into simple vector
coords = [0,0,0,0]
coords[0] = label['shapes'][0]['points'][0][0]
coords[1] = label['shapes'][0]['points'][0][1]
coords[2] = label['shapes'][0]['points'][1][0]
coords[3] = label['shapes'][0]['points'][1][1]

# transform coordinates into albumentations format by dividing by width and height of image
coords = list(np.divide(coords,[640,480,640,480]))
print(coords)

#augment image
augmented = augmentor(image=test, bboxes=[coords], class_labels=['face'])
print(augmented)

#view images and annotations
data_samples = train.as_numpy_iterator()
res = data_samples.next()
fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx in range(4):
    sample_image = res[0][idx]
    sample_coords = res[1][1][idx]

    cv2.rectangle(sample_image,
                  tuple(np.multiply(sample_coords[:2], [120, 120]).astype(int)),
                  tuple(np.multiply(sample_coords[2:], [120, 120]).astype(int)),
                  (255, 0, 0), 2)

    ax[idx].imshow(sample_image)
plt.show()

#used to make test predictions on sample images (not in real time)
test_data = test.as_numpy_iterator()
test_sample = test_data.next()
yhat = facetracker.predict(test_sample[0])
print(yhat)
fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx in range(4):
    sample_image = test_sample[0][idx]
    sample_coords = yhat[1][idx]

    if yhat[0][idx] > 0.9:
        cv2.rectangle(sample_image,
                      tuple(np.multiply(sample_coords[:2], [120, 120]).astype(int)),
                      tuple(np.multiply(sample_coords[2:], [120, 120]).astype(int)),
                      (255, 0, 0), 2)

    ax[idx].imshow(sample_image)