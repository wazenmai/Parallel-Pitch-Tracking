// 1024 thread per block, 1 thread 1 frame

/*
where_is_happiness.wav
        Type  Time(%)      Time     Calls       Avg       Min       Max     Name
GPU activities:   99.94%  4.33811s         1  4.33811s  4.33811s  4.33811s  calculate_pitch

walking.wav
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
GPU activities:   99.93%  9.98657s         1  9.98657s  9.98657s  9.98657s  calculate_pitch
*/


#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

#include <cuda.h>

// WAVE file header format
struct HEADER {
	unsigned char riff[4];						// RIFF string
	unsigned int overall_size	;				// overall size of file in bytes
	unsigned char wave[4];						// WAVE string
	unsigned char fmt_chunk_marker[4];			// fmt string with trailing null char
	unsigned int length_of_fmt;					// length of the format data
	unsigned int format_type;					// format type. 1-PCM, 3- IEEE float, 6 - 8bit A law, 7 - 8bit mu law
	unsigned int channels;						// no.of channels
	unsigned int sample_rate;					// sampling rate (blocks per second)
	unsigned int byterate;						// SampleRate * NumChannels * BitsPerSample/8
	unsigned int block_align;					// NumChannels * BitsPerSample/8
	unsigned int bits_per_sample;				// bits per sample, 8- 8bits, 16- 16 bits etc
	unsigned char data_chunk_header [4];		// DATA string or FLLR string
	unsigned int data_size;						// NumSamples * NumChannels * BitsPerSample/8 - size of the next chunk that will be read
};

#define PI 3.14159265

FILE* ptr;
struct HEADER header;

unsigned char buffer4[4];
unsigned char buffer2[2];
unsigned char buffer1;

int max_freq = 1000;
int min_freq = 40;
const int T = 1024;

void print_wav_header(FILE *ptr);
char* seconds_to_time(float raw_seconds);
long get_num_samples();
long get_size_of_each_sample();
float get_duration_in_seconds();
bool is_valid_sample(long size_of_each_sample);
double frame2volume(int* frame, int frame_size);
int get_median_of_frame(int* frame, int frame_size);

void normalizeAndScale(int audioData[], int length, int desiredBits);
void medianFilter(int* input, int* output, int length, int windowSize);

__global__ void calculate_pitch(int* data, int* pitch, int frame_size, int num_frames, int n1, int n2, int fs) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int start = (bid * T + tid) * frame_size;
    // frame_size: # of samples in a frame
    // num_frames: # of frames
    double acf = 0.0;
    double max_acf = 0.0;
    int max_acf_index = 0;
    for (int shift = n1; shift < n2; shift++) {
        double out = 0.0;
        double deminator = 0.0;
        for (int j = 0; j < frame_size - shift; j++) {
            out += data[start + j] * data[start + j + shift];
            deminator += data[start + j] * data[start + j] + data[start + j + shift] * data[start + j + shift];
        }
        acf = (deminator > 0) ? 2 * out / deminator : 0;
        if (acf > max_acf) {
            max_acf = acf;
            max_acf_index = shift;
        }
    }
    double freq = (double)fs / max_acf_index;
    double semitone = 12 * log2(freq / 440) + 69;
    // if (bid == 100)
        // printf("freq=%f, semitone=%f\n", freq, semitone);
    pitch[bid * T + tid] = round(semitone);
}

int main(int argc, char **argv) {
    int read = 0;

    if (argc < 2) {
        printf("Usage: %s filename.wav\n", argv[0]);
        exit(0);
    }

    ptr = fopen(argv[1], "rb");
    if (ptr == NULL) {
        printf("Error opening file\n");
        exit(1);
    }

    // NOTE: 1. read header parts
    print_wav_header(ptr);

    // NOTE: 2. calculate information about file
    long num_samples = get_num_samples(); // 1: print, 0: don't print
    long size_of_each_sample = get_size_of_each_sample();
    float duration_in_seconds = get_duration_in_seconds();

    if (!is_valid_sample(size_of_each_sample)) {
        exit(1);
    }

    long bytes_in_each_channel = (size_of_each_sample / header.channels);
    char data_buffer[size_of_each_sample];
    long i = 0;
    long low_limit = 0l;
    long high_limit = 0l;
    
    switch (header.bits_per_sample) {
        case 8:
            low_limit = -128;
            high_limit = 127;
            break;
        case 16:
            low_limit = -32768;
            high_limit = 32767;
            break;
        case 32:
            low_limit = -2147483648;
            high_limit = 2147483647;
            break;
    }
    printf("nn.Valid range for data values : %ld to %ld \n", low_limit, high_limit);

    // NOTE: 3. Read data chunks
    int* data[header.channels];
    for (i = 0; i < header.channels; i++) {
        data[i] = (int*)malloc(sizeof(int) * num_samples);
    }
    double avg = 0.0;
    for (i = 0; i < num_samples; i++) {
        read = fread(data_buffer, sizeof(data_buffer), 1, ptr);
        if (read == 1) {
            unsigned int xchannels = 0;
            int data_in_channel = 0;
            int offset = 0; // move the offset for every iteration in the loop below
            for (xchannels = 0; xchannels < header.channels; xchannels++) {
                // NOTE: 4. convert data from little endian to big endian based on bytes in each sample
                if (bytes_in_each_channel == 4) {
                    data_in_channel = data_buffer[offset] | 
                                      (data_buffer[offset + 1] << 8) | 
                                      (data_buffer[offset + 2] << 16) | 
                                      (data_buffer[offset + 3] << 24);
                } else if (bytes_in_each_channel == 2) {
                    data_in_channel = data_buffer[offset] | (data_buffer[offset + 1] << 8);
                } else if (bytes_in_each_channel == 1) {
                    data_in_channel = data_buffer[offset];
                }

                offset += bytes_in_each_channel;
                if (data_in_channel < low_limit || data_in_channel > high_limit) {
                    printf("**value out of range\n");
                    data[xchannels][i] = 0;
                } else {
                    data[xchannels][i] = data_in_channel;
                }
                avg = (avg + (data[xchannels][i] - avg) / (i + 1));
            }
        } else {
            printf("Error reading file. %d bytes\n", read);
            break;
        }
    }
    fclose(ptr);

    // NOTE: 4. Frame blocking
    const int frame_duration = 100; // 100ms
    const int frame_size = header.sample_rate * frame_duration / 1000;
    const int num_frames = num_samples / frame_size;
    printf("frame_size=%d, num_of_frames=%d\n", frame_size, num_frames);
    // NOTE: 5. Calculate volume
    double* volume = (double*)malloc(sizeof(double) * num_frames);
    double volume_threasold = 0.0;
    for (i = 0; i < num_frames; i++) {
        volume[i] = frame2volume(data[0] + i * frame_size, frame_size);
        volume_threasold = fmax(volume_threasold, volume[i]);
        // printf("frame %ld: volume=%f\n", i, volume[i]);
    }
    volume_threasold *= 0.1;
    printf("volume_threasold=%f\n", volume_threasold);
    // NOTE: 6. Calculate max and min frequency for pitch
    int fs = header.sample_rate;
    int n1 = floor(fs / max_freq);
    int n2 = ceil(fs / min_freq);
    printf("n1=%d, n2=%d\n", n1, n2);

    // ADD: Allocate device global memory and copy data to device
    // Input - Only one channel
    int* data_device;
    cudaMalloc((void**)&data_device, sizeof(int) * num_samples);
    cudaMemcpy(data_device, data[0], sizeof(int) * num_samples, cudaMemcpyHostToDevice);
    
    // Output
    int* pitch = (int*)malloc(sizeof(int) * num_frames);
    int* pitch_device;
    cudaMalloc((void**)&pitch_device, sizeof(int) * num_frames);

    // ADD: Claim the thread for gpu
    int T = 1024;
    int M = (num_frames + T - 1) / T;
    calculate_pitch<<<M, T>>>(data_device, pitch_device, frame_size, num_frames, n1, n2, fs);

    // ADD: Copy data back to host
    cudaMemcpy(pitch, pitch_device, sizeof(int) * num_frames, cudaMemcpyDeviceToHost);
    cudaFree(data_device);
    cudaFree(pitch_device);

    // Smooth the pitch
    int window_size = 10;
    int* pitch_smooth = (int*)malloc(sizeof(int) * num_frames);
    medianFilter(pitch, pitch_smooth, num_frames, window_size);

    for (i=0; i < num_frames; i++) {
        if (volume[i] < volume_threasold) {
            printf("0, ");
        } else {
            printf("%d, ", pitch_smooth[i]);
        }
    }
    // for (i = 0; i < num_frames; i++) {
    //     printf("%d, ", pitch[i]);
    // }

    // free data
    for (i = 0; i < header.channels; i++) {
        free(data[i]);
    }
    free(pitch);
    return 0;
}


void print_wav_header(FILE *ptr) {
	if (ptr == NULL) return;

	int read = 0;

	// read header parts
	// fread: the position of the file pointer is updated automatically after the read operation, 
	// so that successive fread() functions read successive file records.
	read = fread(header.riff, sizeof(header.riff), 1, ptr);
	printf("(1-4): %s \n", header.riff); 

	read = fread(buffer4, sizeof(buffer4), 1, ptr);
	printf("%u %u %u %u\n", buffer4[0], buffer4[1], buffer4[2], buffer4[3]);

	// convert little endian to big endian 4 byte int
	header.overall_size  = buffer4[0] | 
						(buffer4[1]<<8) | 
						(buffer4[2]<<16) | 
						(buffer4[3]<<24);

	printf("(5-8) Overall size: bytes:%u, Kb:%u \n", header.overall_size, header.overall_size/1024);

	read = fread(header.wave, sizeof(header.wave), 1, ptr);
	printf("(9-12) Wave marker: %s\n", header.wave);

	read = fread(header.fmt_chunk_marker, sizeof(header.fmt_chunk_marker), 1, ptr);
	printf("(13-16) Fmt marker: %s\n", header.fmt_chunk_marker);

	read = fread(buffer4, sizeof(buffer4), 1, ptr);
	printf("%u %u %u %u\n", buffer4[0], buffer4[1], buffer4[2], buffer4[3]);

	// convert little endian to big endian 4 byte integer
	header.length_of_fmt = buffer4[0] |
							(buffer4[1] << 8) |
							(buffer4[2] << 16) |
							(buffer4[3] << 24);
	printf("(17-20) Length of Fmt header: %u \n", header.length_of_fmt);

	read = fread(buffer2, sizeof(buffer2), 1, ptr); printf("%u %u \n", buffer2[0], buffer2[1]);

	header.format_type = buffer2[0] | (buffer2[1] << 8);
	char format_name[10] = "";
	if (header.format_type == 1)
	strcpy(format_name,"PCM"); 
	else if (header.format_type == 6)
	strcpy(format_name, "A-law");
	else if (header.format_type == 7)
	strcpy(format_name, "Mu-law");

	printf("(21-22) Format type: %u %s \n", header.format_type, format_name);

	read = fread(buffer2, sizeof(buffer2), 1, ptr);
	printf("%u %u \n", buffer2[0], buffer2[1]);

	header.channels = buffer2[0] | (buffer2[1] << 8);
	printf("(23-24) Channels: %u \n", header.channels);

	read = fread(buffer4, sizeof(buffer4), 1, ptr);
	printf("%u %u %u %u\n", buffer4[0], buffer4[1], buffer4[2], buffer4[3]);

	header.sample_rate = buffer4[0] |
						(buffer4[1] << 8) |
						(buffer4[2] << 16) |
						(buffer4[3] << 24);

	printf("(25-28) Sample rate: %u\n", header.sample_rate);

	read = fread(buffer4, sizeof(buffer4), 1, ptr);
	printf("%u %u %u %u\n", buffer4[0], buffer4[1], buffer4[2], buffer4[3]);

	header.byterate  = buffer4[0] |
						(buffer4[1] << 8) |
						(buffer4[2] << 16) |
						(buffer4[3] << 24);
	printf("(29-32) Byte Rate: %u , Bit Rate:%u\n", header.byterate, header.byterate*8);

	read = fread(buffer2, sizeof(buffer2), 1, ptr);
	printf("%u %u \n", buffer2[0], buffer2[1]);

	header.block_align = buffer2[0] |
					(buffer2[1] << 8);
	printf("(33-34) Block Alignment: %u \n", header.block_align);

	read = fread(buffer2, sizeof(buffer2), 1, ptr);
	printf("%u %u \n", buffer2[0], buffer2[1]);

	header.bits_per_sample = buffer2[0] |
					(buffer2[1] << 8);
	printf("(35-36) Bits per sample: %u \n", header.bits_per_sample);
    
    // ADD: For stupid format of soo and LIST INFO
    read = fread(buffer2, sizeof(buffer2), 1, ptr);
    printf("buffer2: %u %u\n", buffer2[0], buffer2[1]);
    if (buffer2[0] == 100 && buffer2[1] == 97) { // d, a
        // Normal format
        header.data_chunk_header[0] = 'd';
        header.data_chunk_header[1] = 'a';
        read = fread(buffer2, sizeof(buffer2), 1, ptr);
        header.data_chunk_header[2] = 't';
        header.data_chunk_header[3] = 'a';
    } else if (buffer2[0] == 76 && buffer2[1] == 73) { // L, I
        // LIST INFO format
        while (true) {
            read = fread(buffer2, sizeof(buffer2), 1, ptr);
            // printf("buffer2: %u %u\n", buffer2[0], buffer2[1]);
            if (buffer2[0] == 100 && buffer2[1] == 97) { // d, a
                break;
            }
        }
        header.data_chunk_header[0] = 'd';
        header.data_chunk_header[1] = 'a';
        read = fread(buffer2, sizeof(buffer2), 1, ptr);
        header.data_chunk_header[2] = 't';
        header.data_chunk_header[3] = 'a';
    }

	// read = fread(header.data_chunk_header, sizeof(header.data_chunk_header), 1, ptr);
	printf("(37-40) Data Marker: %s \n", header.data_chunk_header);
    printf("data_chunk_header: %u %u %u %u\n", header.data_chunk_header[0], header.data_chunk_header[1], header.data_chunk_header[2], header.data_chunk_header[3]);
	read = fread(buffer4, sizeof(buffer4), 1, ptr);
	printf("%u %u %u %u\n", buffer4[0], buffer4[1], buffer4[2], buffer4[3]);

	header.data_size = buffer4[0] |
				(buffer4[1] << 8) |
				(buffer4[2] << 16) | 
				(buffer4[3] << 24 );
	printf("(41-44) Size of data chunk: %u \n", header.data_size);
}

long get_num_samples() {
	// print data_size, num_channels, bits_per_sample, block_align
	printf("data_size=%u, num_channels=%u, bits_per_sample=%u, block_align=%u \n", header.data_size, header.channels, header.bits_per_sample, header.block_align);
	long num_samples = (8 * header.data_size) / (header.channels * header.bits_per_sample);
	printf("Number of samples:%lu \n", num_samples);
	return num_samples;
}

long get_size_of_each_sample() {
	long size_of_each_sample = (header.channels * header.bits_per_sample) / 8;
	printf("Size of each sample:%ld bytes\n", size_of_each_sample);
	return size_of_each_sample;
}

float get_duration_in_seconds() {
	float duration_in_seconds = (float) header.overall_size / header.byterate;
	printf("Approx.Duration in seconds=%f\n", duration_in_seconds);
	return duration_in_seconds;
}

bool is_valid_sample(long size_of_each_sample) {
	if (header.format_type != 1) return false;
	
	long bytes_in_each_channel = (size_of_each_sample / header.channels);
	// make sure that the bytes-per-sample is completely divisible by num.of channels
	if ((bytes_in_each_channel  * header.channels) != size_of_each_sample) {
		printf("Error: %ld x %ud <> %ld\n", bytes_in_each_channel, header.channels, size_of_each_sample);
		return false;
	}
	return true;
}

char* seconds_to_time(float raw_seconds) {
	char *hms;
	int hours, hours_residue, minutes, seconds, milliseconds;
	hms = (char*) malloc(100);

	sprintf(hms, "%f", raw_seconds);

	hours = (int) raw_seconds/3600;
	hours_residue = (int) raw_seconds % 3600;
	minutes = hours_residue/60;
	seconds = hours_residue % 60;
	milliseconds = 0;

	// get the decimal part of raw_seconds to get milliseconds
	char *pos;
	pos = strchr(hms, '.');
	int ipos = (int) (pos - hms);
	char decimalpart[15];
	memset(decimalpart, ' ', sizeof(decimalpart));
	strncpy(decimalpart, &hms[ipos+1], 3);
	milliseconds = atoi(decimalpart);	


	sprintf(hms, "%d:%d:%d.%d", hours, minutes, seconds, milliseconds);
	return hms;
}

double* note_envelope(double* time_vec) {
    double max_amplitude = 0.9;
    double b = 0.015; // peak_time
    double width = 0.05;
    // double k = 10;
    // double period = 0.05;
    // double exponent = 20;

    // The function = c*t/(t^2+a*t+b^2), with the peak at [b, c/(a+2*b)]
    // Let z=a+4*b, then 50% height occurs at (z-sqrt(z*z-4*b^2))/2 and (z+sqrt(z*z-4*b^2))/2.
    double a = sqrt(width * width + 4 * b * b) - 4 * b; // 0.523
    double c = max_amplitude * (a + 2 * b); // 0.4977
    // get the size of time_vec
    int size = 1000; // TODO: get the size of time_vec
    double* envelope = (double*)malloc(sizeof(double) * size);
    for (int i = 0; i < size; i++) {
        double t = time_vec[i];
        envelope[i] = c * t / (t * t + a * t + b * b); // time=0.022, envelope = 0.089
    }
    return envelope;
}

double frame2volume(int* frame, int frame_size) {
    double volume = 0;
    // abssum
    int median = get_median_of_frame(frame, frame_size);
    // print median
    // printf("median=%f\n", median);
    for (int i = 0; i < frame_size; i++) {
        volume += abs(frame[i] - median);
    }
    return volume;
}

int compare (const void * a, const void * b) {
    return ( *(double*)a - *(double*)b );
}

int int_compare (const void * a, const void * b) {
    return ( *(int*)a - *(int*)b );
}

int get_median_of_frame(int* frame, int frame_size) {
    int* frame_copy = (int*)malloc(sizeof(int) * frame_size);
    memcpy(frame_copy, frame, sizeof(int) * frame_size);
    qsort(frame_copy, frame_size, sizeof(int), compare);
    free(frame_copy);
    return frame_copy[frame_size / 2];
}

void normalizeAndScale(int audioData[], int length, int desiredBits) {
    // Calculate the maximum value for a 16-bit integer
    int max16Bit = (1 << 15);  // Equivalent to (2 ** 16) / 2

    // Calculate the maximum value for the desired bit level
    int maxDesired = (1 << (desiredBits - 1));  // Equivalent to (2 ** desiredBits) / 2

    // Loop through all values, normalize them to 1, then scale to the new max value
    for (int i = 0; i < length; i++) {
        double normalisedSample = (double)audioData[i] / max16Bit;
        int scaledSample = (int)(normalisedSample * maxDesired);
        audioData[i] = scaledSample;
    }
}

void medianFilter(int* input, int* output, int length, int windowSize) {
    int halfWindow = windowSize / 2;
    int windowValues[windowSize];

    for (int i = 0; i < length; i++) {
        // Determine the range of indices for the window
        int start = (i - halfWindow < 0) ? 0 : i - halfWindow;
        int end = (i + halfWindow >= length) ? length - 1 : i + halfWindow;

        // Copy the values within the window to a temporary array
        for (int j = start; j <= end; j++) {
            windowValues[j - start] = input[j];
        }

        // Sort the temporary array (e.g., using bubble sort for simplicity)
        qsort(windowValues, windowSize, sizeof(int), int_compare);

        // Set the output value to the median of the sorted window
        output[i] = windowValues[windowSize / 2];
    }
}

/* GPU properties
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  2048
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
*/
