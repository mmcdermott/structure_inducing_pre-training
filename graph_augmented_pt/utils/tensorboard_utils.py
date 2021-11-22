import struct, tensorboard.compat.proto.event_pb2 as event_pb2, glob, os, shutil
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def read(data):
    header = struct.unpack('Q', data[:8])

    # crc_hdr = struct.unpack('I', data[:4])

    event_str = data[12:12+int(header[0])] # 8+4
    data = data[12+int(header[0])+4:]
    return data, event_str

def read_tensorboard_log(filepath):
    with open(filepath, mode='rb') as f: data = f.read()

    events = []
    event_streams = defaultdict(list)
    steps = defaultdict(list)
    times = defaultdict(list)

    while data:
        data, event_str = read(data)
        event = event_pb2.Event()
        event.ParseFromString(event_str)
        events.append(event)

        if not event.HasField('summary'): continue

        for value in event.summary.value:
            if not value.HasField('simple_value'): continue

            event_streams[value.tag].append(value.simple_value)
            steps[value.tag].append(event.step)
            times[value.tag].append(event.wall_time)

    return events, event_streams, steps, times

def get_best_epoch(sample_dir, remove_others=True):
    best_epoch_file = sample_dir / 'best_epoch.txt'
    if best_epoch_file.exists():
        with open(best_epoch_file, 'r') as f:
            return int(f.readline().strip())

    DEFAULT_SIZE_GUIDANCE = {'scalars': 0}
    glob_arg = str(sample_dir / 'lightning_logs/*/events*')
    tb_log = glob.glob(glob_arg)
    assert len(tb_log) == 1, f'Expected only one events file! Found {tb_log}.'
    tb_log = tb_log[0]

    event_acc = EventAccumulator(tb_log, DEFAULT_SIZE_GUIDANCE)
    event_acc.Reload()
    tags = event_acc.Tags()["scalars"]
    assert 'Val/loss' in tags, "Couldn't find Val/loss!"

    event_list = event_acc.Scalars('Val/loss')
    values = list(map(lambda x: x.value, event_list))

    min_value = 1e50
    for i, value in enumerate(values):
        if value < min_value:
            min_value = value
            index = i

    # Clean up other epochs, since they take a lot of space.
    if remove_others:
        epochs_dir = sample_dir / 'epochs'
        epochs_files = os.listdir(epochs_dir)
        for e in epochs_files:
            if int(e) != index:
                shutil.rmtree(epochs_dir / e)

    with open(best_epoch_file, 'w') as f:
        f.write(str(index))
    return index
