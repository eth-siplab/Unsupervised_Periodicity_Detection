
def plot_save_signals(self, signal, time, path):
    duration, sampling_rate = 8, 100 
    # Calculate time axis
    if time is None:
        time = torch.linspace(0, duration, int(duration * sampling_rate), dtype=torch.float32, requires_grad=True).cuda(self.args.cuda)
        time = time.squeeze()
    plt.figure()
    if isinstance(signal, tuple):
        plt.plot(time.cpu().detach().numpy(), signal[0].squeeze().cpu().detach().numpy())
        plt.plot(time.cpu().detach().numpy(), signal[1].squeeze().cpu().detach().numpy())
    else: 
        plt.plot(time.cpu().detach().numpy(), signal.squeeze().cpu().detach().numpy())
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(path)
    plt.grid(True)
    savepath = path + '.png'
    # Save the plot as a PNG image
    plt.savefig(savepath)
    plt.cla()

def find_intervals(self, signal):
    intervals = []
    start_index = None

    for i in range(len(signal)):
        if signal[i] > 0:
            if start_index is None:
                start_index = i
        elif signal[i] == 0 and start_index is not None:
            intervals.append((start_index, i))
            start_index = None

    if start_index is not None:
        intervals.append((start_index, len(signal)))

    # Ignore the first and last intervals
    intervals = intervals[1:-1]

    # Calculate the average interval length
    interval_lengths = [end_index - start_index for start_index, end_index in intervals]
    avg_interval_length = int(np.mean(interval_lengths))

    # Truncate or pad intervals to have the average length
    for i in range(len(intervals)):
        start_index, end_index = intervals[i]
        interval_length_diff = end_index - start_index - avg_interval_length

        if interval_length_diff > 0:
            # Truncate interval if longer
            intervals[i] = (start_index, end_index - interval_length_diff)
        elif interval_length_diff < 0:
            # Pad interval if shorter
            intervals[i] = (start_index, end_index - interval_length_diff)

    return intervals

def segment_signal(self, signal, intervals):
    segment = torch.zeros([intervals[0][1]-intervals[0][0]]).cuda(self.args.cuda)
    # Iterate over intervals, ignoring the first and last interval
    for start_index, end_index in intervals:
        segment = torch.add(signal[start_index:end_index],segment)         
    segment = segment/len(intervals)

    return segment