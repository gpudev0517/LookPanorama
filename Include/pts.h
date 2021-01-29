#ifndef PTS
#define PTS

#include <QStringList>

#define MIN_FILE_SIZE       1024
#define START_SYM           "#"
#define VALUE_SYM           "-"
#define BLOCK_SYM           ":"
#define VALUE_EXP           "#-"
#define BLOCK_EXP           ":\n"
#define COMMENT_EXP         "# "
#define ROOT_BLOCK          "Root"
#define IMAGE_VAL           "imgfile"
#define IMAGE_KEY           "input images->imgfile"
#define SHARE_VAL           "dummyimage"
#define SHARE_KEY           "input images->dummyimage"
#define CAMERA_EXP          "o "
#define EXPOSURE_VAL        "exposureparams"
#define EXPOSURE_KEY		"input images->exposureparams"

typedef enum {
    NONE_VAL = -1,
    BOOL_VAL,
    NUMBER_VAL,
    STRING_VAL,
    STRUCTURE_VAL,
} TYPEOFVAL;

typedef struct {
    TYPEOFVAL type;
    QStringList value;
    int nbVal;
    QMap<QString, float> camParams;
	QMap<QString, int> camParamRefs; // Equal with camera #n
} PTSVAL;

#endif // PTS

