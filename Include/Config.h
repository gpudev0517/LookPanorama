

#ifndef CONFIG_H
#define CONFIG_H

// FPS statistics queue lengths
#define PROCESSING_FPS_STAT_QUEUE_LENGTH    32
#define CAPTURE_FPS_STAT_QUEUE_LENGTH       32
#define STITCH_FPS_STAT_QUEUE_LENGTH		32
#define OCULUS_FPS_STAT_QUEUE_LENGTH		75
#define STREAMING_FPS_STAT_QUEUE_LENGTH     32

// Image buffer size
#define DEFAULT_IMAGE_BUFFER_SIZE           1
// Drop frame if image/frame buffer is full
#define DEFAULT_DROP_FRAMES                 false
// Thread priorities
#define DEFAULT_CAP_THREAD_PRIO             QThread::NormalPriority
#define DEFAULT_PROC_THREAD_PRIO            QThread::HighPriority

// IMAGE PROCESSING
// Smooth
#define DEFAULT_SMOOTH_TYPE                 0 // Options: [BLUR=0,GAUSSIAN=1,MEDIAN=2]
#define DEFAULT_SMOOTH_PARAM_1              3
#define DEFAULT_SMOOTH_PARAM_2              3
#define DEFAULT_SMOOTH_PARAM_3              0
#define DEFAULT_SMOOTH_PARAM_4              0
// Dilate
#define DEFAULT_DILATE_ITERATIONS           1
// Erode
#define DEFAULT_ERODE_ITERATIONS            1
// Flip
#define DEFAULT_FLIP_CODE                   0 // Options: [x-axis=0,y-axis=1,both axes=-1]
// Canny
#define DEFAULT_CANNY_THRESHOLD_1           10
#define DEFAULT_CANNY_THRESHOLD_2           00
#define DEFAULT_CANNY_APERTURE_SIZE         3
#define DEFAULT_CANNY_L2GRADIENT            false

// Performance Update
#define REMOVE_PROCESSING_THREAD_IN_CAMERAVIEW

// Snapshot detector
#define SNAPSHOT_MAX_STRENGTH 1000
#define SNAPSHOT_THRESHOLD_STRENGTH 0.35f
#define SNAPSHOT_DURATION 3

// Checkboard information
#define CHECKBOARD_WIDTH 8
#define CHECKBOARD_HEIGHT 6

#endif // CONFIG_H
