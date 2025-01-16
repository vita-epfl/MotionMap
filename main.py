import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np; np.random.seed(42)

from vis import visualize_skeleton_compare 

item = 400

local_maxima = np.load(f"data/sup_{item}_local_maxima.npy")
local_maxima_aux = np.load(f"data/sup_{item}_local_maxima_aux.npy")
y_mm = np.load(f"data/sup_{item}_y_mm.npy")
y_hm = np.load(f"data/sup_{item}.npy")

y_mm = np.concatenate([np.zeros((y_mm.shape[0], 125, 1, 1, 3)), y_mm], axis=-2)

# z_hm = np.load("data/z_hm_train.npy")
# action_colors = np.load("data/action_colors_train.npy")


x_local_max = local_maxima[:,1]
y_local_max = local_maxima[:,0]

x_aux = local_maxima_aux[:,1]
y_aux = local_maxima_aux[:,0]


def remove_empty_spaces(image, dimention=None):
    #remoce a column is it is all white and the neighbering columns are also white:
    i = 5
    while i < image.shape[1]-20:
        if np.all(image[:,i,0:3] == 255) and  np.all(image[:,i+3,0:3] == 255): #np.all(image[:,i-1,0:3] == 255) and
            image = np.delete(image, i, axis=1)
        i += 1
        
    if dimention is None:
        return image
    
    if image.shape[1] != dimention[1]:
        if image.shape[1] < dimention[1]:
            image = np.pad(image, ((0,0), (dimention[1]-image.shape[1],0), (0,0)), 'constant', constant_values=255)
        else:
            image = image[:, :dimention[1], ...]
    
    return image

temp = visualize_skeleton_compare(y_mm[0,25:], y_mm[0,25:], dataset=None, save_path=None, string=None, return_array=True)
plt.close()
temp = remove_empty_spaces(temp)
image_dim = temp.shape
arr = np.empty((y_mm.shape[0],image_dim[0]//2,image_dim[1],image_dim[2]))
print(arr.shape)
for i in range(y_mm.shape[0]):
    temp = visualize_skeleton_compare(y_mm[i,25:], y_mm[i,25:],
            dataset=None, save_path=None, string=None, return_array=True)[image_dim[0]//2:, :, :] #cut the image in half
    
    arr[i] = remove_empty_spaces(temp, dimention=image_dim)/255
    plt.close()
      
inp_array = visualize_skeleton_compare(y_mm[i,:25], y_mm[i,:25], dataset=None, save_path=None, string=None, return_array=True) 
inp_array = inp_array/255
inp_array = inp_array[:inp_array.shape[0]//2, : , : ] 

    
fig = plt.figure()
gs = fig.add_gridspec(2, 1, height_ratios=[1, 9])  # Adjust the height ratios as needed

ax0 = fig.add_subplot(gs[0])
ax = fig.add_subplot(gs[1])


ax0.imshow(inp_array[:,:-70])
# plt.scatter(z_hm[:, 1]+5, z_hm[:, 0], s=0.25, c=action_colors, alpha=0)
ax.imshow(y_hm, alpha=0.75)

ax0.set_xticks([])
ax0.set_yticks([])
ax0.set_title('Input Sequence')
ax.set_title('MotionMap')

x = np.concatenate([x_local_max, x_aux], axis=0)
y = np.concatenate([y_local_max, y_aux], axis=0)
print(x.shape, y.shape)
line, = ax.plot(x,y, ls="", marker="o",  color='white', markersize=0)

line_loca_lmax, = ax.plot(x_local_max, y_local_max, ls="", marker="x",  color='red', markersize=5)
line_aux, = ax.plot(x_aux, y_aux, ls="", marker="o", color='black', markersize=0.25)


im = OffsetImage(arr[0,:,:], zoom= 0.5)#25)
xybox=(50., 50.)
ab = AnnotationBbox(im, (0,0), xybox=xybox, xycoords='data',
        boxcoords="offset points",  pad=0.3,  arrowprops=dict(arrowstyle="->"))
ax.add_artist(ab)
ab.set_visible(False)

def hover(event):
    if line.contains(event)[0]:
        ind = line.contains(event)[1]["ind"][0]
        w,h = fig.get_size_inches()*fig.dpi
        ws = (event.x > w/2.)*-1 + (event.x <= w/2.) 
        hs = (event.y > h/2.)*-1 + (event.y <= h/2.)
        ab.xybox = (xybox[0]*ws, xybox[1]*hs)
        ab.set_visible(True)
        ab.xy =(x[ind], y[ind])
        
        im.set_data(arr[ind]) 
    else:
        #if the mouse is not over a scatter point
        ab.set_visible(False)
    fig.canvas.draw_idle()

fig.canvas.mpl_connect('motion_notify_event', hover)           
plt.show()