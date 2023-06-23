
import ezomero
import sys
import matplotlib.pyplot as plt

def connect(user, password):
    HOST = 'ome2.hpc.sussex.ac.uk' #change if different
    port=4064
    conn = ezomero.connect(user=user,password=password,group='',host=HOST,port=4064,secure=True)
    if conn: print('Connection successful')
    else: print('Unsuccessful')
    return conn

def progressbar(it, prefix="", size=60, out=sys.stdout): # Python3.3+
    count = len(it)
    def show(j):
        x = int(size*j/count)
        print("{}[{}{}] {}/{}".format(prefix, "#"*x, "."*(size-x), j, count), 
                end='\r', file=out, flush=True)
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    print("\n", flush=True, file=out)

def extract_channel(data_images,channel):
    data_images_channel = []
    for i in range(len(data_images)):
        data_images_channel.append(data_images[i][0][0][:,:,channel])
    return data_images_channel

if __name__ == '__main__':
    conn = connect(user='rz200',password='omeroreset')

    plate = 1237
    image_ids = ezomero.get_image_ids(conn,plate=plate)

    print('In plate',plate,'we have',len(image_ids),'images')

    data_images = []
    for i in progressbar(range(len(image_ids)), "Computing: ", 40):
        data_images.append(ezomero.get_image(conn, image_ids[i])[1])
    
    data_images_one = extract_channel(data_images, 0)
    data_images_two = extract_channel(data_images, 1)

    plt.subplot(2,3,1)
    plt.imshow(data_images_one[0])
    plt.subplot(2,3,2)
    plt.imshow(data_images_one[4])
    plt.subplot(2,3,3)
    plt.imshow(data_images_one[10])

    plt.subplot(2,3,4)
    plt.imshow(data_images_one[0])
    plt.subplot(2,3,5)
    plt.imshow(data_images_one[4])
    plt.subplot(2,3,6)
    plt.imshow(data_images_one[10])

    plt.show()