import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import io


skeleton_h36m = [[0,1],[1,2],[2,3],[0,7],[7,8],[8,9],[9,10],[8,14],[14,15],[15,16],  
                [0,4],[4,5],[5,6],[8,11],[11,12],[12,13] ] #left

skeleton_amass = [[0,2],[2,5],[5,8],[8,11],
                    [0,3],[3,6],[6,9],[9,12],[12,15],
                    [9,14], [14,17], [17,19], [19,21],
                    [0,1], [1,4], [4,7], [7,10], #left
                    [9,13], [13,16],[16,18],[18,20]] #left
                    
color_pairs = [
        ("lightseagreen", "turquoise"),            # 0 ("palevioletred" if j<l_j else "pink")
        ("palevioletred" , "pink")  ,              # 1
        ("lightskyblue", "steelblue"), 
        ("thistle", "mediumorchid"),                   
        ("khaki", "gold"),                         # 5
        ("lightgreen", "limegreen"),               # 6
        ("sandybrown", "chocolate"),               # 7
        ("lightpink", "palevioletred"),            # 8
        ("powderblue", "lightslategray"),          # 9
        ("peachpuff", "darkorange"),               # 10
        ("lavender", "rebeccapurple"),             # 11 - purplish
        ("plum", "purple"),                        # 12 - Purplish
        ("violet", "darkviolet"),                  # 13 - purplish
        ("orchid", "darkorchid"),                  # 14 - Purplish
        ("palegoldenrod", "goldenrod"),            # 15
        ("azure", "navy"),                         # 16 - Contrast with a deep color
        ("mistyrose", "firebrick"),                # 17
        ("lemonchiffon", "khaki"),                 # 18
        ("lightcyan", "cyan"),                     # 19
        ("lavender", "blueviolet")                 # 20 - purplish
    ]

#color_pairs = colors = [
#    ((100/255, 140/255, 140/255), (40/255, 80/255, 80/255)),   # Dark Teal
#    ((130/255, 180/255, 210/255), (70/255, 120/255, 150/255)),   # Blueish Tone
#    ((210/255, 160/255, 180/255), (150/255, 100/255, 120/255))  # Pink 
#]

def visualize_seq(seq, dataset, save_path, string):
    _,_,j,_ = seq.shape
    seq_17kp = dataset.recover_landmarks(seq, rrr=True, fill_root=True)
    _plot_seq(seq_17kp, save_path, string)


def visualize_skeleton_compare(seq1, seq2, dataset, save_path, string, return_array=False):
    
    n_columns = 9 if seq1.shape[0]==25 else 8
    
    skip = min(seq1.shape[0], seq2.shape[0])//n_columns + 1 #use this if you only want one row!
    
    seq1 = seq1[::skip,:,:,:]
    seq2 = seq2[::skip,:,:,:]
    
    seq1_end_frame = seq1.shape[0]
    seq = np.concatenate((seq1, seq2), axis=0)

    if dataset is not None:
        seq = dataset.recover_landmarks(seq, rrr=True, fill_root=True)
    else:
        seq = seq
    
    n_rows = seq.shape[0]//n_columns + int(bool(seq1_end_frame%n_columns>0)) + int(bool(seq2.shape[0]%n_columns>0))
    fig = plt.figure(figsize=(n_columns,n_rows))
    # fig.tight_layout()
    j = 0
    for i in range(seq.shape[0]):
        ax = fig.add_subplot(n_rows, n_columns, j+1, projection='3d')
                
        frame_number = i*skip if i<seq1_end_frame else (i-seq1_end_frame)*skip
        _plot_skeleton(ax, seq[i, 0], label=f"pose|{i}", title=f"frame {frame_number}", seq_number = 1 if i<seq1_end_frame else 2)
        
        if i == seq1_end_frame - 1: #it means it was the last from the first sequence and till here, i=j
            while (j+1)%n_columns != 0: #till the next location is not the first of the next row
                j+=1
        j+=1

    plt.subplots_adjust(wspace=0, hspace=0)
    #ONLY KEEP THIS FOR HOVER!
    if return_array:
        fig_aux = plt.figure(figsize=(1,2))
        for i in range(2):
            ax = fig_aux.add_subplot(2, 1,i+1, projection='3d')
            for k in range(seq1.shape[0]):
                _plot_skeleton(ax,seq[k+i*(seq1_end_frame),0], label="",title="",seq_number = i+1)


    if not return_array:
        os.makedirs(os.path.join(save_path, "plots"), exist_ok=True)
    
    plt.subplots_adjust(wspace=0, hspace=0)
    
    if return_array:

        # fig.canvas.draw()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        image = Image.open(buf)
        image_array = np.array(image) #shpe: (~200(row), ~800(column), 4(RGBA))
 
        #new:
        buf.close()
        # fig_aux.canvas.draw()
        buf_aux = io.BytesIO()
        fig_aux.savefig(buf_aux, format='png', bbox_inches='tight', pad_inches=0)
        image_aux = Image.open(buf_aux)
        image_aux_array = np.array(image_aux)
        image_array = np.concatenate((image_array, image_aux_array), axis=1)

        # image = Image.fromarray(image_array, 'RGBA')
        # image.save('filename.png')
        
        buf_aux.close()
        plt.close(fig)
        plt.close(fig_aux)
        return image_array
 
    else:
        # Split the string into parts
        parts = string.split('/')
        
        # Create the directory path
        directory = os.path.join(save_path, "plots", *parts[:-1])
        
        # Create the directories
        os.makedirs(directory, exist_ok=True)
        
        # Save the figure
        plt.savefig(os.path.join(directory, "{}.png".format(parts[-1])), bbox_inches='tight', pad_inches=0)
        plt.close()


def _setup_axes(ax):
    ax.set_xlabel('X') 
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    
    ax.grid(False)
    ax.set_axis_off()
        
    ax.axes.set_xlim3d(left=-0.5, right=0.5) 
    ax.axes.set_ylim3d(bottom=-0.5, top=0.5) 
    ax.axes.set_zlim3d(bottom=-0.5 , top=0.5 ) 
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


def _plot_skeleton(ax, data, label, title, seq_number=1):
    xdata, ydata, zdata = data.T
    ax.scatter(xdata, ydata, zdata, color="black", s=1.25) #"turquoise" label=label,  0.5

    n_joints = xdata.shape[0]
    if n_joints == 22:
        skeleton = skeleton_amass
    else: 
        skeleton = skeleton_h36m
    l_j = 10 if seq_number==1 else 13
    for j in range(n_joints-1):
        color_indx = seq_number % len(color_pairs) - 1
        
        #COLOR:
        #if seq_number>2:
        #    color_indx = 2
        
        selected_pair = color_pairs[color_indx]
        color =  selected_pair[0] if j < l_j else selected_pair[1] #("lightseagreen" if j<l_j else "turquoise") if seq_number%2==1 else ("palevioletred" if j<l_j else "pink")
        ax.plot(xdata[ skeleton[j]], ydata[skeleton[j]], zdata[skeleton[j]] , color = color, linewidth=2, alpha=0.9) #1.5
        
    # ax.set_title(title)
    # ax.view_init(elev=120, azim=-60)
    _setup_axes(ax)



def visualize_skeletons(tensor,tensor_y,  name="skeletons"):
    KeyPoints_from3d = [0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27]
    
    x_f = tensor.reshape(-1,16,3) #.clone().detach().cpu().numpy()
    x_f = np.concatenate((np.zeros((x_f.shape[0],1,3)), x_f), axis=1) #np.append(x_f, np.zeros((x_f.shape[0],1,3)),axis=1)
    
    y_f = tensor_y.reshape(-1,16,3) #.clone().detach().cpu().numpy()
    y_f = np.concatenate((np.zeros((y_f.shape[0],1,3)), y_f), axis=1) # np.append(y_f, np.zeros((y_f.shape[0],1,3)),axis=1)
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1,2,1, projection='3d')
    
    for i in range(x_f.shape[0]):
        xdata = x_f[i].T[0]
        ydata = x_f[i].T[1]
        zdata = x_f[i].T[2]
        ax.scatter(xdata,ydata,zdata, label = f"pose|{i}" )
        for j in range(16):
            ax.plot(xdata[ skeleton[j]], ydata[skeleton[j]], zdata[skeleton[j]])
        ax.set_title("frame {}".format(i))
        ax.view_init(elev=120, azim=-60)
        _setup_axes(ax)
        # plt.show()
        
    ax.scatter(0,0,0, color='red', marker='*', label='origin')
        
    ax2 = fig.add_subplot(1,2,2, projection='3d')
    for i in range(y_f.shape[0]):
        xdata = y_f[i].T[0]
        ydata = y_f[i].T[1]
        zdata = y_f[i].T[2]
        ax2.scatter(xdata,ydata,zdata, label = f"pose|{i}" )
        for j in range(16):
            ax2.plot(xdata[ skeleton[j]], ydata[skeleton[j]], zdata[skeleton[j]])
        ax2.set_title("frame {}".format(i))
        # ax2.view_init(elev=120, azim=-60)
        _setup_axes(ax2)
        # plt.show()
    
    #make the path if does not exist:
    if not os.path.exists("./my_plots"):
        os.makedirs("./my_plots")
    
    plt.savefig("./my_plots/"+name+".png")
    # plt.savefig("./my_plots/"+name+"_0.pdf") 
    plt.close(fig)


def _plot_seq(seq, save_path, string):
    seq_shape = seq.shape #(100 or 25, 1, 17, 3)
    
    fig = plt.figure(figsize=(20, 20))

    for i in range(seq_shape[0]):
        ax = fig.add_subplot(seq_shape[0]//5, 5, i+1, projection='3d')
        _plot_skeleton(ax, seq[i, 0], label=f"pose|{i}", title=f"frame {i}")

    # Save the entire grid as a single image
    os.makedirs(os.path.join(save_path, "plots"), exist_ok=True)
    plt.savefig(os.path.join(save_path, "plots/{}.png".format(string)))
    plt.close(fig)


