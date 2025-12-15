from mcap.reader import SeekingReader
from mcap.writer import Writer
from mcap.writer import Schema, Channel
from mcap_ros2.decoder import Decoder
from munch import munchify
import numpy as np
import struct
import collections
from tqdm import tqdm
import re


class McapMsg(object):
    def __init__(self, item):
        self.pub_ns = item[2].publish_time
        self.log_ns = item[2].log_time
        self.seq = item[2].sequence
        self.item = item
        self._data = None

    def schema(self):
        return self.item[0].data.decode('utf-8')

    @property
    def topic(self):
        return self.item[1].topic

    @property
    def data(self):
        if self._data is None:
            self._data = McapDB.decode(self.item[0], self.item[2])
        return self._data

    @property
    def pub_ts(self):
        return self.pub_ns * 1e-9

    @property
    def log_ts(self):
        return self.log_ns * 1e-9

class McapDB(object):

    '''Initialize the database object with the file path to the mcap file'''
    def __init__(self, file_path, is_deep=False):
        self.file_path = file_path
        self._fp = open(self.file_path, 'rb')
        self._reader = SeekingReader(self._fp)
        self._summary = self._reader.get_summary()
        self.start_log_ts = None
        self.start_pub_ts = None
        self.start_header_ts = None
        self._topics = self._indexing(is_deep)

    def close(self):
        self._fp.close()

    def __len__(self):
        return self._summary.statistics.message_count

    def _indexing(self, is_deep):
        '''Create indices to topics and their information'''
        topics = dict()
        for item in tqdm(self._reader.iter_messages()):
            channel = item[1]
            topic = channel.topic
            message = item[2]
            log_time = message.log_time
            pub_time = message.publish_time
            if self.start_log_ts is None:
                self.start_log_ts = log_time * 1e-9
            if self.start_pub_ts is None:
                self.start_pub_ts = pub_time * 1e-9
            if topic in topics:
                info = topics[topic]
                begin_time, end_time = info.lifespan
                count = info.count
                topics[topic].lifespan = (min(begin_time, pub_time), max(end_time, log_time))
                topics[topic].count = count + 1
            else:
                topics[topic] = munchify({
                    'lifespan': (pub_time, log_time + 1),
                    'count': 1,
                    'log_time': []
                })
            if is_deep:
                topics[topic].log_time.append(log_time)
        return munchify(topics)

    def generator(self, every=1, skip=0, max_count=None, topics=None):
        count = 0
        ingested_count = 0
        count_per_topic = collections.defaultdict(lambda: 0)
        for item in self._reader.iter_messages():
            count += 1
            if count <= skip:
                continue
            if max_count is not None and ingested_count >= max_count:
                break
            ingested_count += 1
            if every == 1 or count_per_topic[topic] % every == 0:
                channel = item[1]
                topic = channel.topic
                if topics is None or topic in topics:
                    yield McapMsg(item)
                    count_per_topic[topic] += 1

    def topics(self):
        '''Return a list of topic names in the bag'''
        return list(self._topics.keys())

    def has_topic(self, topic):
        return topic in self.topics()
    
    def topic_info(self, topic_name):
        '''Return the topic object with the given topic name'''
        return self._topics[topic_name]

    def topic(self, topic_name):
        info = self.topic_info(topic_name)
        return self.get([topic_name], info.lifespan[0], info.lifespan[1])

    def sample(self, topics, start_time_ns, end_time_ns, window_ns, count=None, interval_ns=None):
        if count is not None:
            interval_ns = (end_time_ns - start_time_ns) // count
        assert(interval_ns is not None)
        half_window_ns = window_ns // 2
        return [self.get(topics, sample_ns - half_window_ns, sample_ns + half_window_ns)
                for sample_ns in range(start_time_ns, end_time_ns, interval_ns)]

    def get(self, topics, start_time_ns, end_time_ns):
        items = self._reader.iter_messages(
                topics, start_time_ns, end_time_ns)
        return list(map(lambda item: McapMsg(item), items))

    @staticmethod
    def decode(schema, message):
        decoder = Decoder()
        return decoder.decode(schema=schema, message=message)

    @staticmethod
    def get_image(ros_img) -> np.ndarray:
        """
        Convert a dict of ImageMessage to a numpy array.
        """
        if ros_img is None:
            return

        encoding = ros_img.encoding
        height = ros_img.height
        width = ros_img.width
        step = ros_img.step
        data = ros_img.data

        if encoding == 'rgb8':
            if step != width * 3:
                raise Exception(f'Error extracting rgb8. Step should be {width*3}, but is {step}')
            np_img = np.zeros([height, width, 3], dtype=np.uint8)
            for i in range(height):
                start_idx = i * step
                end_idx = (i + 1) * step
                np_img[i, :, 0] = np.frombuffer(data[start_idx : end_idx : 3], dtype=np.uint8)
                np_img[i, :, 1] = np.frombuffer(data[start_idx+1 : end_idx : 3], dtype=np.uint8)
                np_img[i, :, 2] = np.frombuffer(data[start_idx+2 : end_idx : 3], dtype=np.uint8)
        elif encoding == 'bgr8':
            if(step != width*3):
                raise Exception(f'Error extracting bgr8. Step should be {width*3}, but is {step}')
            np_img = np.zeros([height, width, 3], dtype=np.uint8)
            for i in range(height):
                start_idx = i * step
                end_idx = (i + 1)*step
                np_img[i, :, 2] = np.frombuffer(data[start_idx : end_idx : 3], dtype=np.uint8)
                np_img[i, :, 1] = np.frombuffer(data[start_idx + 1 : end_idx : 3], dtype=np.uint8)
                np_img[i, :, 0] = np.frombuffer(data[start_idx + 2 : end_idx : 3], dtype=np.uint8)
        elif encoding == "32FC1":
            np_img = np.reshape(np.frombuffer(data, dtype=np.float32), (height, width))
        elif encoding == "mono8":
            np_img = np.reshape(np.frombuffer(data, dtype=np.uint8), (height, width))
        elif encoding == "32FC3":
            np_img = np.reshape(np.frombuffer(data, dtype=np.float32), (height, width, 3))
        elif encoding == 'bgra8':
            if(step != width*4):
                raise Exception(f'Error extracting bgra. Step should be {width*4}, but is {step}')
            np_img = np.zeros([height, width, 4], dtype=np.uint8)
            for i in range(height):
                start_idx = i * step
                end_idx = (i + 1)*step
                # Convert to rgba
                np_img[i, :, 2] = np.frombuffer(data[start_idx : end_idx : 4], dtype=np.uint8)
                np_img[i, :, 1] = np.frombuffer(data[start_idx + 1 : end_idx : 4], dtype=np.uint8)
                np_img[i, :, 0] = np.frombuffer(data[start_idx + 2 : end_idx : 4], dtype=np.uint8)
                np_img[i, :, 3] = np.frombuffer(data[start_idx + 3 : end_idx : 4], dtype=np.uint8)
        else:
            raise Exception(f'Image encoding \'{encoding}\' is not supported')
        return np_img

    def create_reversed_mcap(self, output_file_path):
        """
        Create a new MCAP file with messages in reverse order, preserving intervals.
        """
        # Read all messages
        messages = list(self._reader.iter_messages())
        
        # Calculate the time span
        if not messages:
            print("No messages to process.")
            return

        first_message_time = messages[0][2].log_time
        last_message_time = messages[-1][2].log_time
        total_time_span = last_message_time - first_message_time

        # Reverse the order of messages
        reversed_messages = reversed(messages)
        
        # Open a new file to write the reversed messages
        with open(output_file_path, 'wb') as output_fp:
            writer = Writer(output_fp)
            writer.start()
            # Assuming you have a way to get schema and channel information
            schema_map = {}
            channel_map = {}

            for item in reversed_messages:
                schema, channel, message = item

                # Calculate the new log and publish times
                original_log_time = message.log_time
                new_log_time = first_message_time + (last_message_time - original_log_time)
                original_publish_time = message.publish_time
                new_publish_time = first_message_time + (last_message_time - original_publish_time)

                # Register schema if not already registered
                if schema.id not in schema_map:
                    schema_map[schema.id] = writer.register_schema(
                        name=schema.name,
                        encoding=schema.encoding,
                        data=schema.data
                    )

                # Register channel if not already registered
                if channel.id not in channel_map:
                    channel_map[channel.id] = writer.register_channel(
                        topic=channel.topic,
                        message_encoding=channel.message_encoding,
                        schema_id=schema_map[schema.id]
                    )

                # Write the message with adjusted timestamps
                writer.add_message(
                    channel_id=channel_map[channel.id],
                    log_time=new_log_time,
                    publish_time=new_publish_time,
                    data=message.data  # Use the original message data
                )
            writer.finish()
