import os
import argparse
import yaml
from mcap.reader import make_reader
from mcap.writer import Writer
from io import BytesIO


class MCAPSplitter:
    def __init__(self, input_dir, max_size, output_dir, suffix):
        self.input_dir = input_dir
        self.max_size = max_size
        self.output_dir = output_dir
        self.suffix = suffix
        self.base_name = os.path.basename(input_dir.rstrip('/'))
        self.mcap_file, self.metadata_file = self._find_files()
        self.metadata = self._load_metadata()
        self.schemas_written = {}
        self.channels_written = {}

    def _find_files(self):
        mcap_file = None
        metadata_file = None
        
        for file_name in os.listdir(self.input_dir):
            if file_name.endswith('.mcap'):
                mcap_file = os.path.join(self.input_dir, file_name)
            elif file_name.endswith('metadata.yaml'):
                metadata_file = os.path.join(self.input_dir, file_name)
        
        if not mcap_file or not metadata_file:
            raise FileNotFoundError("Both .mcap and metadata.yaml files must be present in the input directory.")
        
        return mcap_file, metadata_file

    def _load_metadata(self):
        with open(self.metadata_file, 'r') as meta_infile:
            return yaml.safe_load(meta_infile)

    def _initialize_chunk_writer(self):
        chunk_data = BytesIO()
        mcap_writer = Writer(chunk_data)
        mcap_writer.start()
        return mcap_writer, chunk_data

    def _finalize_chunk_writer(self, mcap_writer, chunk_data, chunk_filename, chunk_metadata, chunk_message_count):
        mcap_writer.finish()

        with open(chunk_filename, 'wb') as chunk_file:
            chunk_file.write(chunk_data.getvalue())

        chunk_metadata["rosbag2_bagfile_information"]["relative_file_paths"] = [os.path.basename(chunk_filename)]
        chunk_metadata["rosbag2_bagfile_information"]["files"] = [{
            "path": os.path.basename(chunk_filename),
            "starting_time": chunk_metadata["rosbag2_bagfile_information"]["starting_time"],
            "duration": chunk_metadata["rosbag2_bagfile_information"]["duration"],
            "message_count": chunk_message_count
        }]

    def _write_metadata_chunk(self, metadata_chunk_filename, chunk_metadata):
        with open(metadata_chunk_filename, 'w') as meta_outfile:
            yaml.safe_dump(chunk_metadata, meta_outfile)

    def _initialize_chunk_metadata(self):
        return {
            "rosbag2_bagfile_information": {
                "version": self.metadata["rosbag2_bagfile_information"]["version"],
                "storage_identifier": self.metadata["rosbag2_bagfile_information"]["storage_identifier"],
                "compression_format": self.metadata["rosbag2_bagfile_information"]["compression_format"],
                "compression_mode": self.metadata["rosbag2_bagfile_information"]["compression_mode"],
                "duration": {"nanoseconds": 0},
                "starting_time": self.metadata["rosbag2_bagfile_information"]["starting_time"],
                "message_count": 0,
                "topics_with_message_count": []
            }
        }

    def _update_chunk_metadata(self, chunk_metadata, channel, message, record_size):
        topic_name = channel.topic
        topic_found = False
        for topic_entry in chunk_metadata["rosbag2_bagfile_information"]["topics_with_message_count"]:
            if topic_entry["topic_metadata"]["name"] == topic_name:
                topic_entry["message_count"] += 1
                topic_found = True
                break
        
        if not topic_found:
            chunk_metadata["rosbag2_bagfile_information"]["topics_with_message_count"].append({
                "topic_metadata": {
                    "name": topic_name,
                    "type": channel.message_encoding,
                    "serialization_format": channel.message_encoding,
                    "offered_qos_profiles": channel.metadata.get("offered_qos_profiles", "")
                },
                "message_count": 1
            })

        chunk_metadata["rosbag2_bagfile_information"]["message_count"] += 1
        chunk_metadata["rosbag2_bagfile_information"]["duration"]["nanoseconds"] += record_size

    def _manage_schema_and_channel(self, mcap_writer, schema, channel):
        if schema.id not in self.schemas_written:
            schema_id = mcap_writer.register_schema(name=schema.name, encoding=schema.encoding, data=schema.data)
            self.schemas_written[schema.id] = schema_id
        else:
            schema_id = self.schemas_written[schema.id]

        if channel.id not in self.channels_written:
            channel_id = mcap_writer.register_channel(schema_id=schema_id, topic=channel.topic, message_encoding=channel.message_encoding)
            self.channels_written[channel.id] = channel_id
        else:
            channel_id = self.channels_written[channel.id]

        return schema_id, channel_id

    def _prepare_chunk_paths(self, chunk_index):
        chunk_folder = os.path.join(self.output_dir, f"{self.base_name}{self.suffix}_{chunk_index}")
        os.makedirs(chunk_folder, exist_ok=True)
        chunk_filename = os.path.join(chunk_folder, f"{self.base_name}{self.suffix}_{chunk_index}_0.mcap")
        metadata_chunk_filename = os.path.join(chunk_folder, f"{self.base_name}{self.suffix}_{chunk_index}_metadata.yaml")
        return chunk_folder, chunk_filename, metadata_chunk_filename

    def split(self):
        with open(self.mcap_file, 'rb') as infile:
            mcap_reader = make_reader(infile)
            
            current_size = 0
            chunk_index = 1
            chunk_metadata = self._initialize_chunk_metadata()
            chunk_message_count = 0

            mcap_writer, chunk_data = self._initialize_chunk_writer()

            for schema, channel, message in mcap_reader.iter_messages():
                record_size = len(message.data)

                if current_size + record_size > self.max_size:
                    chunk_folder, chunk_filename, metadata_chunk_filename = self._prepare_chunk_paths(chunk_index)

                    self._finalize_chunk_writer(mcap_writer, chunk_data, chunk_filename, chunk_metadata, chunk_message_count)
                    self._write_metadata_chunk(metadata_chunk_filename, chunk_metadata)

                    chunk_index += 1
                    current_size = 0
                    chunk_metadata = self._initialize_chunk_metadata()
                    chunk_message_count = 0
                    self.schemas_written.clear()
                    self.channels_written.clear()
                    mcap_writer, chunk_data = self._initialize_chunk_writer()

                schema_id, channel_id = self._manage_schema_and_channel(mcap_writer, schema, channel)
                mcap_writer.add_message(channel_id=channel_id, log_time=message.log_time, data=message.data, publish_time=message.publish_time)
                
                current_size += record_size
                chunk_message_count += 1
                self._update_chunk_metadata(chunk_metadata, channel, message, record_size)

            if current_size > 0:
                chunk_folder, chunk_filename, metadata_chunk_filename = self._prepare_chunk_paths(chunk_index)
                self._finalize_chunk_writer(mcap_writer, chunk_data, chunk_filename, chunk_metadata, chunk_message_count)
                self._write_metadata_chunk(metadata_chunk_filename, chunk_metadata)


class MCAPTopicSplitter(MCAPSplitter):
    def __init__(self, input_dir, max_size, output_dir, suffix, split_topic):
        super().__init__(input_dir, max_size, output_dir, suffix)
        self.split_topic = split_topic

    def split(self):
        with open(self.mcap_file, 'rb') as infile:
            mcap_reader = make_reader(infile)
            
            current_size = 0
            chunk_index = 1
            chunk_metadata = self._initialize_chunk_metadata()
            chunk_message_count = 0

            mcap_writer, chunk_data = self._initialize_chunk_writer()

            for schema, channel, message in mcap_reader.iter_messages():
                record_size = len(message.data)

                # Split based on the topic
                if channel.topic == self.split_topic:
                    chunk_folder, chunk_filename, metadata_chunk_filename = self._prepare_chunk_paths(chunk_index)

                    self._finalize_chunk_writer(mcap_writer, chunk_data, chunk_filename, chunk_metadata, chunk_message_count)
                    self._write_metadata_chunk(metadata_chunk_filename, chunk_metadata)

                    chunk_index += 1
                    current_size = 0
                    chunk_metadata = self._initialize_chunk_metadata()
                    chunk_message_count = 0
                    self.schemas_written.clear()
                    self.channels_written.clear()
                    mcap_writer, chunk_data = self._initialize_chunk_writer()

                schema_id, channel_id = self._manage_schema_and_channel(mcap_writer, schema, channel)
                mcap_writer.add_message(channel_id=channel_id, log_time=message.log_time, data=message.data, publish_time=message.publish_time)
                
                current_size += record_size
                chunk_message_count += 1
                self._update_chunk_metadata(chunk_metadata, channel, message, record_size)

            if current_size > 0:
                chunk_folder, chunk_filename, metadata_chunk_filename = self._prepare_chunk_paths(chunk_index)
                self._finalize_chunk_writer(mcap_writer, chunk_data, chunk_filename, chunk_metadata, chunk_message_count)
                self._write_metadata_chunk(metadata_chunk_filename, chunk_metadata)


class MCAPPubTsSplitter(MCAPSplitter):
    def __init__(self, input_dir, output_dir, suffix, start_pub_ts, end_pub_ts):
        super().__init__(input_dir, None, output_dir, suffix)
        self.start_pub_ts = start_pub_ts
        self.end_pub_ts = end_pub_ts

    def split(self):
        with open(self.mcap_file, 'rb') as infile:
            mcap_reader = make_reader(infile)
            
            current_size = 0
            chunk_index = 1
            chunk_metadata = self._initialize_chunk_metadata()
            chunk_message_count = 0

            mcap_writer, chunk_data = self._initialize_chunk_writer()

            for schema, channel, message in mcap_reader.iter_messages():
                record_size = len(message.data)

                # Split based on the pub_ts
                if self.start_pub_ts <= message.publish_time * 1e-9 <= self.end_pub_ts:
                    schema_id, channel_id = self._manage_schema_and_channel(mcap_writer, schema, channel)
                    mcap_writer.add_message(channel_id=channel_id, log_time=message.log_time, data=message.data, publish_time=message.publish_time)
                    
                    current_size += record_size
                    chunk_message_count += 1
                    self._update_chunk_metadata(chunk_metadata, channel, message, record_size)

            if current_size > 0:
                chunk_folder, chunk_filename, metadata_chunk_filename = self._prepare_chunk_paths(chunk_index)
                self._finalize_chunk_writer(mcap_writer, chunk_data, chunk_filename, chunk_metadata, chunk_message_count)
                self._write_metadata_chunk(metadata_chunk_filename, chunk_metadata)


def main():
    parser = argparse.ArgumentParser(description='Split a large MCAP file based on max size, specific topics, or publication timestamps.')
    parser.add_argument('input_dir', type=str, help='Path to the directory containing the input MCAP file and metadata.yaml.')
    parser.add_argument('output_dir', type=str, help='Directory to store the resulting chunks and metadata.')
    parser.add_argument('suffix', type=str, help='Suffix to add to the chunk filenames.')
    parser.add_argument('--max_size', type=int, default=None, help='Maximum size for each chunk in bytes.')
    parser.add_argument('--split_topic', type=str, default=None, help='Topic name to split the file.')
    parser.add_argument('--start_pub_ts', type=int, default=None, help='Start publication timestamp to split the file.')
    parser.add_argument('--end_pub_ts', type=int, default=None, help='End publication timestamp to split the file.')

    args = parser.parse_args()

    if args.start_pub_ts and args.end_pub_ts:
        splitter = MCAPPubTsSplitter(args.input_dir, args.output_dir, args.suffix, args.start_pub_ts, args.end_pub_ts)
    elif args.split_topic:
        splitter = MCAPTopicSplitter(args.input_dir, args.max_size, args.output_dir, args.suffix, args.split_topic)
    else:
        splitter = MCAPSplitter(args.input_dir, args.max_size, args.output_dir, args.suffix)

    splitter.split()


if __name__ == '__main__':
    main()

