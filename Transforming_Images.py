def 32x128_grayscale():
    directory = 'words/'
    for sub1 in os.listdir(directory):
        for sub2 in os.listdir(directory + sub1 + '/'):
            for files in os.listdir(directory + sub1 + '/' + sub2 + '/'):
                path = directory + sub1 + '/' + sub2 + '/' + files
                photo = load_img(path,color_mode="grayscale", target_size=(32, 128))
                photo.save(path,'PNG')





