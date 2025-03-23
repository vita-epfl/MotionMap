import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import io

from matplotlib.animation import FuncAnimation
import matplotlib


class Visualizer:
    """
    A class for visualizing skeleton data in 3D space.

    Attributes:
        save_path (str): Path to save visualizations.
        dataset_name (str): Name of the dataset (e.g., 'Human36M', 'AMASS').
        dataset (object): Dataset object for recovering landmarks.
        ax_limits (dict): Axis limits for 3D plots.
        color_pairs (list): List of color pairs for skeleton visualization.
        skeleton (list): List of joint connections for the skeleton.
        numbers (list): Predefined list of numbers for visualization purposes.
    """

    def __init__(self, save_path, dataset=None):
        """
        Initialize the Visualizer object.

        Args:
            save_path (str): Path to save visualizations.
            dataset (object, optional): Dataset object for recovering landmarks.
        """
        self.save_path = save_path
        self.dataset_name = dataset.dataset if dataset is not None else None
        self.dataset = dataset
        self.ax_limits = {"x":{"min":-0.5, "max":0.5}, "y":{"min":-0.5, "max":0.5}, "z":{"min":-0.5, "max":0.5}} 
        
        self.color_pairs = [
            ("lightseagreen", "turquoise"),            # 0 
            ("palevioletred" , "pink")  ,              # 1
            ("lightskyblue", "steelblue"),             # 2
            ("thistle", "mediumorchid"),               # 3  
            ("lightcoral", "red"),                     # 4  
            ("khaki", "gold"),                         # 5
            ("lightgreen", "limegreen"),               # 6
            ("sandybrown", "chocolate"),               # 7
            ("lightpink", "palevioletred"),            # 8
            ("powderblue", "lightslategray"),          # 9
            ("peachpuff", "darkorange"),               # 10
            ("lavender", "rebeccapurple"),             # 11 
            ("plum", "purple"),                        # 12 
            ("violet", "darkviolet"),                  # 13 
            ("orchid", "darkorchid"),                  # 14 
            ("palegoldenrod", "goldenrod"),            # 15
            ("azure", "navy"),                         # 16 
            ("mistyrose", "firebrick"),                # 17
            ("lemonchiffon", "khaki"),                 # 18
            ("lightcyan", "cyan"),                     # 19
            ("lavender", "blueviolet")                 # 20 
        ]
        
        self.current_color = None
        
        skeleton_h36m = [[0,1],[1,2],[2,3],[0,7],[7,8],[8,9],[9,10],[8,14],[14,15],[15,16],  
                [0,4],[4,5],[5,6],[8,11],[11,12],[12,13] ] #left

        skeleton_amass = [[0,2],[2,5],[5,8],[8,11],
                            [0,3],[3,6],[6,9],[9,12],[12,15],
                            [9,14], [14,17], [17,19], [19,21],
                            [0,1], [1,4], [4,7], [7,10], #left
                            [9,13], [13,16],[16,18],[18,20]] #left
        
        if self.dataset_name == "Human36M":
            self.skeleton = skeleton_h36m
        elif self.dataset_name == "AMASS":
            self.skeleton = skeleton_amass
        else:
            self.skeleton = None
            
        self.numbers = [ 448, 5051, 4970, 531, 3373, 6, 154, 198, 323, 356, 407,
                460, 654, 834, 901, 1028, 1057, 1162, 1269, 1359,
                1416, 1487, 1540, 1566, 1611, 1643, 1687, 1878, 1985, 2025,
                2081, 2086, 2306, 2335, 2363, 2478, 2494, 2511, 2633, 2740,
                2796, 2842, 3003, 3033, 3148, 3235, 3264, 3338, 3388, 3569, 3686,
                3691, 3765, 3899, 4046, 413, 4334, 4475, 4552, 4581, 4680, 4851, 4862,
                4956, 5032, 5038, 5141, 5162]                    
        
    def _plot_skeleton(self, ax, data, label, title, seq_number=1):
        """
        Plot a single skeleton in 3D space.

        Args:
            ax (matplotlib.axes._subplots.Axes3DSubplot): 3D subplot axis.
            data (numpy.ndarray): Skeleton data of shape (N, 3).
            label (str): Label for the plot.
            title (str): Title of the plot.
            seq_number (int): Sequence number for color selection.
        """
        xdata, ydata, zdata = data.T
        ax.scatter(xdata, ydata, zdata, color="black", s=1.25) 
        n_joints = xdata.shape[0]
        skeleton = self.skeleton
        l_j = 10 if seq_number==1 else 13
        for j in range(n_joints-1):
            color_indx = seq_number % len(self.color_pairs) - 1
            selected_pair = self.color_pairs[color_indx]
            color =  selected_pair[0] if j < l_j else selected_pair[1] #("lightseagreen" if j<l_j else "turquoise") if seq_number%2==1 else ("palevioletred" if j<l_j else "pink")
            ax.plot(xdata[ skeleton[j]], ydata[skeleton[j]], zdata[skeleton[j]] , color = color, linewidth=2, alpha=0.9) #1.5
            
        # ax.set_title(title)
        # ax.view_init(elev=120, azim=-60)
        self._setup_axes(ax)
        
    def _setup_axes(self, ax):
        """
        Configure the 3D plot axes.

        Args:
            ax (matplotlib.axes._subplots.Axes3DSubplot): 3D subplot axis.
        """
        ax.set_xlabel('X') 
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        
        ax.grid(False)
        ax.set_axis_off()
            
        ax.axes.set_xlim3d(left=self.ax_limits["x"]["min"], right=self.ax_limits["x"]["max"])
        ax.axes.set_ylim3d(bottom=self.ax_limits["y"]["min"], top=self.ax_limits["y"]["max"])
        ax.axes.set_zlim3d(bottom=self.ax_limits["z"]["min"], top=self.ax_limits["z"]["max"])
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
            
        
    def visualize_skeleton_compare_multi(self, sequences, string, return_array=False, project_under=False, title=None):
        """
        Visualize multiple skeleton sequences in a grid layout.

        Args:
            sequences (list): List of skeleton sequences to visualize.
            string (str): File name or identifier for saving the visualization.
            return_array (bool, optional): Whether to return the visualization as an image array.
            project_under (bool, optional): Whether to project one sequence under another.
            title (str, optional): Title for the visualization.
        """
        if not sequences:
            raise ValueError("No sequences provided")
        
        # Determine the number of columns based on the first sequence's joint count
        n_columns = 9 if sequences[0].shape[0] == 25 else 8
        total_frames = sum(seq.shape[0] for seq in sequences)
        
        # Calculate skip rates for each sequence to limit them to a single row per sequence if necessary
        skip = max([seq.shape[0] for seq in sequences]) // n_columns + 1
        
        # Subsample sequences based on calculated skip
        sequences_subsampled = [seq[::skip, :, :, :] for seq in sequences]
        sequences_subsampled = [self.dataset.recover_landmarks(seq, rrr=True, fill_root=True)
                                for seq in sequences_subsampled]
        
        # data := (BatchSize, SegmentLength, NumPeople, Landmarks, Dimensions)
        # or data := (BatchSize, NumSamples, DiffusionSteps, SegmentLength, NumPeople, Landmarks, Dimensions)
        # the idea is that it does not matter how many dimensions are before NumPeople, Landmarks, Dimension => always working right
       
        
        # Determine the total number of rows needed
        n_rows = sum((seq.shape[0] - 1) // n_columns + 1 for seq in sequences_subsampled)
        fig = plt.figure(figsize=(n_columns, n_rows))
        
        current_frame = 0
        for seq_idx, seq in enumerate(sequences_subsampled):
            for frame_idx in range(seq.shape[0]):
                ax = fig.add_subplot(n_rows, n_columns, current_frame + 1, projection='3d')
                
                if project_under:
                    under_seq = sequences_subsampled[1]
                    self._plot_skeleton(ax, under_seq[frame_idx, 0, :, :], label=f"pose|{current_frame}", title=f"frame {current_frame}", seq_number= 1)

                self._plot_skeleton(ax, seq[frame_idx, 0, :, :], label=f"pose|{current_frame}", title=f"frame {current_frame}", seq_number=seq_idx + 1)
                
                current_frame += 1
            # Adjust for any empty spaces in the last row of each sequence
            while current_frame % n_columns != 0:
                fig.add_subplot(n_rows, n_columns, current_frame + 1)  # Add empty subplots
                plt.axis('off')
                current_frame += 1

        plt.subplots_adjust(wspace=0, hspace=0)
        
        if return_array:
            fig_aux = plt.figure(figsize=(1, n_rows))
            for i in range(len(sequences)):
                ax = fig_aux.add_subplot(n_rows, 1, i + 1, projection='3d')
                for k in range(sequences[i].shape[0]):
                    self._plot_skeleton(ax, sequences[i][k, 0], label="", title="", seq_number=i + 1)
                    
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            buf.seek(0)
            image = Image.open(buf)
            image_array = np.array(image)
            buf.close()
            
            buf_aux = io.BytesIO()
            fig_aux.savefig(buf_aux, format='png', bbox_inches='tight', pad_inches=0)
            buf_aux.seek(0)
            image_aux = Image.open(buf_aux)
            image_aux_array = np.array(image_aux)
            buf_aux.close()
            image_array = np.concatenate((image_array, image_aux_array), axis=1)
            plt.close()
            return image_array
        
        else:
            # Split the string into parts
            parts = string.split('/')
            
            # Create the directory path
            directory = os.path.join(self.save_path, "plots", *parts[:-1])
            
            # Create the directories
            os.makedirs(directory, exist_ok=True)
            
            if title is not None:
                assert type(title) == str, "Title must be a string"

                plt.title(title)
                
            # Save the figure
            plt.savefig(os.path.join(directory, "{}.png".format(parts[-1])), bbox_inches='tight', pad_inches=0, dpi=200)
            plt.savefig(os.path.join(directory, "{}.svg".format(parts[-1])), bbox_inches='tight', pad_inches=0)
            print("Saved!", os.path.join(directory, "{}.png".format(parts[-1])))
            plt.close() 
        
    

    def visualize_skeleton_compare_multi_gif(self, sequences, string):
        """
        Create a GIF to visualize multiple skeleton sequences over time.

        Args:
            sequences (list): List of skeleton sequences to visualize.
            string (str): File name or identifier for saving the GIF.
        """
        # Use Agg backend for creating files
        matplotlib.use("Agg") 
        
        if not sequences:
            raise ValueError("No sequences provided")

        if self.dataset is None:   
            sequences_subsampled = sequences
        else:
            sequences_subsampled = [self.dataset.recover_landmarks(seq, rrr=True, fill_root=True) for seq in sequences]
        
        
        max_frames = max(seq.shape[0] for seq in sequences_subsampled)
        min_frames = min(seq.shape[0] for seq in sequences_subsampled)
        
        fig, axes = plt.subplots(1, len(sequences_subsampled), subplot_kw={'projection': '3d'},
                                figsize=(5 * len(sequences_subsampled), 5))

        def update(frame):
            for ax, seq in zip(axes, sequences_subsampled):
                ax.clear()
                if frame < min_frames:
                    seq_number = 1
                else:
                    seq_number = 2

                if frame < seq.shape[0]:
                    # _plot_skeleton(ax, sequences[i][k, 0], label="", title="", seq_number=i + 1)
                    self._plot_skeleton(ax, seq[frame, 0, :, :], label=f"pose|{frame}", title=f"frame {frame}", seq_number=seq_number)
                    # if project_under and frame < sequences_subsampled[1].shape[0]:
                    #     under_seq = sequences_subsampled[1]
                    #     _plot_skeleton(ax, under_seq[frame, 0, :, :], label=f"pose|{frame}", title=f"frame {frame}", seq_number=1)
                elif frame < max_frames:
                    #plot the last frame
                    self._plot_skeleton(ax, seq[-1, 0, :, :], label=f"pose|{frame}", title=f"frame {seq.shape[0]-1}", seq_number=1)
        
        ani = FuncAnimation(fig, update, frames=max_frames, interval=100)
        
        # Save the GIF
        gif_path = os.path.join(self.save_path, "gifs", f"{string}.gif")
        os.makedirs(os.path.dirname(gif_path), exist_ok=True)
        ani.save(gif_path, writer='imagemagick',
                savefig_kwargs={'bbox_inches': 'tight', 'pad_inches': 0})
        plt.close(fig)


    def find_perpendicular_unit_vectors(self, A):
        """
        Given an array A of shape (N, 3), returns an array of unit vectors 
        that are perpendicular to each row of A while keeping the z-component unchanged.

        Parameters:
            A (numpy.ndarray): Input array of shape (N, 3).

        Returns:
            numpy.ndarray: Array of shape (N, 3) containing unit perpendicular vectors.
        """
        B = np.zeros_like(A, dtype=float)
        
        # Extract components
        a, b, c = A[:, 0], A[:, 1], A[:, 2]
        
        # Compute perpendicular vectors
        x_prime = -b
        y_prime = a
        z_prime = c  # Keeping the z-component unchanged

        # Stack and normalize
        B = np.stack((x_prime, y_prime, z_prime), axis=1)
        B /= np.linalg.norm(B, axis=1, keepdims=True)  # Normalize to unit vectors
        return B