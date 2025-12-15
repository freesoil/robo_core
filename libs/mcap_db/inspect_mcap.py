import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np
from cv_bridge import CvBridge
import cv2
from matplotlib.widgets import Cursor
from mcap_db import McapDB
from functools import partial

def inspect_mcap(mcap_path, topics_to_plot=None, image_topics=None, num_images=10):
    db = McapDB(mcap_path)

    # Print all topics in the MCAP file
    topics = db.topics()
    print("Topics in the MCAP file:")
    for topic in topics:
        count = db.topic_info(topic).count
        print(f"- {topic}\t ({count})")

    # Prepare data for plotting
    timestamp_sequences = []
    sequence_labels = []

    if topics_to_plot:
        for topic in topics_to_plot:
            if db.has_topic(topic):
                topic_db = db.topic(topic)
                timestamps = np.array([item.pub_ts for item in topic_db])
                if len(timestamps):
                    timestamp_sequences.append(timestamps)
                    sequence_labels.append(topic)

    image_sequences = []
    timestamps_list = []
    raw_images = []
    raw_labels = []

    if image_topics:
        bridge = CvBridge()
        for topic in image_topics:
            if db.has_topic(topic):
                topic_db = db.topic(topic)
                timestamps = np.array([item.pub_ts for item in topic_db])
                if len(timestamps):
                    sampled_indices = np.linspace(0, len(timestamps) - 1, num_images, dtype=int)
                    images = [bridge.imgmsg_to_cv2(topic_db[i].data, "rgb8") for i in sampled_indices]
                    raw_images.append([bridge.imgmsg_to_cv2(item.data, "rgb8") for item in topic_db])
                    image_sequences.append(images)
                    timestamps_list.append(timestamps[sampled_indices])
                    raw_labels.append(topic)

    # Compute x-axis limits
    all_timestamps = np.concatenate(timestamp_sequences + timestamps_list)
    min_time = min(all_timestamps)
    max_time = max(all_timestamps)
    time_range = max_time - min_time

    xlim = [mdates.date2num(datetime.fromtimestamp(min_time - 0.1 * time_range)),
            mdates.date2num(datetime.fromtimestamp(max_time + 0.1 * time_range))]

    # Plot diagrams in subplots
    plot_combined_diagrams(timestamp_sequences, sequence_labels, image_sequences, timestamps_list, raw_images, raw_labels, xlim=xlim)

    db.close()

def plot_combined_diagrams(timestamp_sequences, sequence_labels, image_sequences, timestamps_list, raw_images, raw_labels, xlim=None):
    num_sequences = len(image_sequences)
    fig, (ax_band, ax_images, ax_hover) = plt.subplots(3, 1, figsize=(12, num_sequences * 3), gridspec_kw={'height_ratios': [1, len(image_sequences), 2]})

    # Plot band diagram
    plot_band_diagram(ax_band, timestamp_sequences, sequence_labels, xlim)

    # Plot image sequences
    plot_vertical_image_sequences(ax_images, image_sequences, timestamps_list, raw_labels, xlim)

    # Prepare hover plot
    ax_hover.axis('off')

    # Create the hover callback
    additional_hover_callback = partial(display_hover_image, raw_images=raw_images, timestamps_list=timestamps_list, labels=raw_labels, ax_hover=ax_hover)

    # Connect the hover event
    ax_band.figure.canvas.mpl_connect('motion_notify_event', lambda event: on_hover(event, ax_band, additional_hover_callback))

    plt.tight_layout()
    plt.show()

def plot_band_diagram(ax, timestamp_sequences, sequence_labels=None, xlim=None):
    lines_data = []

    min_ts = np.inf
    max_ts = -np.inf
    for idx, timestamps in enumerate(timestamp_sequences):
        if len(timestamps):
            # Convert timestamps to datetime format
            datetimes = [datetime.fromtimestamp(ts) for ts in timestamps]
            # Convert datetime to matplotlib date format
            dates = mdates.date2num(datetimes)

            # Create a band for this sequence
            line = ax.vlines(dates, ymin=idx - 0.4, ymax=idx + 0.4, color='tab:blue', alpha=0.7)
            lines_data.append((line, dates, timestamps))
        min_ts = min(timestamps.min(), min_ts)
        max_ts = max(timestamps.max(), max_ts)

    # Set y-axis labels
    if sequence_labels:
        ax.set_yticks(range(len(sequence_labels)))
        ax.set_yticklabels(sequence_labels)
    else:
        ax.set_yticks(range(len(timestamp_sequences)))
        ax.set_yticklabels([f"Sequence {i+1}" for i in range(len(timestamp_sequences))])

    # Set x-axis limits
    if xlim is None:
        xlim = [min_ts, max_ts]

    ax.set_xlim(xlim)
    ax.set_xticks(np.linspace(xlim[0], xlim[1], 6))

    # Format x-axis to display time
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

    ax.tick_params(axis='x', rotation=45)

    # Ensure the x-tick labels are shown
    plt.setp(ax.get_xticklabels(), visible=True)

    # Enable cursor for mouse-over events
    cursor = Cursor(ax, useblit=True, color='red', linewidth=1)

    # Pass the lines data for hover interaction
    ax.lines_data = lines_data

    ax.set_title('Band Diagram of Timestamp Sequences')

def plot_vertical_image_sequences(ax, image_sequences, timestamps_list, labels, xlim=None):
    num_sequences = len(image_sequences)

    base_seq_id = 1
    for seq_idx, (images, timestamps, label) in enumerate(zip(image_sequences, timestamps_list, labels)):
        # Convert timestamps to matplotlib date format
        datetimes = [datetime.fromtimestamp(ts) for ts in timestamps]
        dates = mdates.date2num(datetimes)

        # Define image height and spacing
        image_height = 1  # Adjust height for each image
        y_position = seq_idx + base_seq_id  # Vertical position for the current sequence, with added space to avoid overlap

        if len(dates) > 1:
            avg_spacing = (dates[-1] - dates[0]) / (len(dates) - 1)
        else:
            avg_spacing = 0.1  # Default small spacing if only one image

        for img, date in zip(images, dates):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (100, 100))  # Resize for consistent display size
            extent = [date - avg_spacing / 2, date + avg_spacing / 2, y_position - image_height / 2, y_position + image_height / 2]
            ax.imshow(img, extent=extent, aspect='auto')

    # Set x-axis limits
    if xlim:
        ax.set_xlim(xlim)

    # Formatting the x-axis to display time
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

    ax.tick_params(axis='x', rotation=45)

    # Ensure the x-tick labels are shown
    plt.setp(ax.get_xticklabels(), visible=True)

    # Set y-axis to correspond to sequence indices
    ax.set_yticks(np.arange(base_seq_id, base_seq_id + num_sequences, 1))
    ax.set_yticklabels(labels)
    ax.set_ylim(base_seq_id - image_height / 2, base_seq_id + num_sequences - image_height/2)

def on_hover(event, ax, additional_hover_callback=None):
    if event.inaxes == ax:
        lines_data = getattr(ax, 'lines_data', [])
        for line, dates, timestamps in lines_data:
            if line.contains(event)[0]:  # Check if the cursor is near any of the vertical lines
                idx = np.abs(dates - event.xdata).argmin()
                closest_time = timestamps[idx]
                closest_datetime = datetime.fromtimestamp(closest_time)
                tooltip_text = f"Time: {closest_datetime.strftime('%Y-%m-%d %H:%M:%S')} | Raw Timestamp: {closest_time}"
                ax.set_title(tooltip_text)
                ax.figure.canvas.draw_idle()
                if additional_hover_callback:
                    additional_hover_callback(closest_time)

def display_hover_image(closest_time, raw_images, timestamps_list, labels, ax_hover):
    ax_hover.clear()
    ax_hover.axis('off')

    concatenated_images = []

    for seq_idx, (images, timestamps, label) in enumerate(zip(raw_images, timestamps_list, labels)):
        # Find the closest image in time
        idx = np.abs(timestamps - closest_time).argmin()
        closest_img = images[idx]

        # Convert the image to RGB and resize it
        closest_img = cv2.cvtColor(closest_img, cv2.COLOR_BGR2RGB)
        closest_img = cv2.resize(closest_img, (200, 200))  # Resize to a square for consistency

        # Add a label to the image (optional)
        closest_img = cv2.putText(closest_img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
        timestamp_label = datetime.fromtimestamp(timestamps[idx]).strftime('%Y-%m-%d %H:%M:%S')
        closest_img = cv2.putText(closest_img, timestamp_label, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

        concatenated_images.append(closest_img)

    # Concatenate images vertically
    if concatenated_images:
        concatenated_image = np.hstack(concatenated_images)
        ax_hover.imshow(concatenated_image)
    
    ax_hover.figure.canvas.draw_idle()

def main():
    parser = argparse.ArgumentParser(description="Inspect an MCAP file and plot topic data over time.")
    parser.add_argument('mcap_dir', type=str, help='Path to the directory containing the MCAP file')
    parser.add_argument('--topics', type=str, nargs='+', help='List of topics to plot')
    parser.add_argument('--image_topics', type=str, nargs='+', help='List of image topics to plot in the image bars')
    parser.add_argument('--num_images', type=int, default=10, help='Number of images to sample uniformly for each image topic')

    args = parser.parse_args()
    
    mcap_path = args.mcap_dir
    topics_to_plot = args.topics
    image_topics = args.image_topics
    num_images = args.num_images

    if not os.path.exists(mcap_path):
        print(f"Error: The file {mcap_path} does not exist.")
        return

    inspect_mcap(mcap_path, topics_to_plot, image_topics, num_images)

if __name__ == '__main__':
    main()

