
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "asiosys.h"
#include "asio.h"
#include "asiodrivers.h"

extern AsioDrivers* asioDrivers;
extern bool loadAsioDriver(char *name);

enum
{
    // number of input and outputs supported by the host application
    // you can change these to higher or lower values
    kMaxInputChannels = 32,
    kMaxOutputChannels = 32
};

typedef struct
{
    ASIODriverInfo driver;

    // ASIOInit()
    ASIODriverInfo driverInfo;

    // ASIOGetChannels()
    long           inputChannels;
    long           outputChannels;

    // ASIOGetBufferSize()
    long           minSize;
    long           maxSize;
    long           preferredSize;
    long           granularity;

    // ASIOGetSampleRate()
    ASIOSampleRate sampleRate;

    // ASIOOutputReady()
    bool           postOutput;

    // ASIOGetLatencies ()
    long           inputLatency;
    long           outputLatency;

    // ASIOCreateBuffers ()
    long inputBuffers;	// becomes number of actual created input buffers
    long outputBuffers;	// becomes number of actual created output buffers
    ASIOBufferInfo bufferInfos[kMaxInputChannels + kMaxOutputChannels]; // buffer info's

    // ASIOGetChannelInfo()
    ASIOChannelInfo channelInfos[kMaxInputChannels + kMaxOutputChannels]; // channel info's
    // The above two arrays share the same indexing, as the data in them are linked together
} DriverInfo;

DriverInfo drv;
ASIOCallbacks asioCallbacks;

// Main audio processing callback.
// NOTE: Called on a separate thread from main() thread.
ASIOTime *bufferSwitchTimeInfo(ASIOTime *timeInfo, long index, ASIOBool processNow)
{
    // Buffer size (in samples):
    long buffSize = drv.preferredSize;

    for (int i = 0; i < drv.outputBuffers; ++i)
    {
        if (i >= drv.inputBuffers) continue;

        // Assumes int32 LSB type:
        assert(drv.channelInfos[i].type == ASIOSTInt32LSB);

        ASIOBufferInfo &inBuf  = drv.bufferInfos[i];
        ASIOBufferInfo &outBuf = drv.bufferInfos[i + drv.inputBuffers];

        assert(inBuf.channelNum == outBuf.channelNum);
        assert(inBuf.isInput == ASIOTrue);
        assert(outBuf.isInput == ASIOFalse);

        // Copy input buffer to output buffer:
        memcpy (outBuf.buffers[index], inBuf.buffers[index], buffSize * 4);
    }

    if (drv.postOutput)
        ASIOOutputReady();

    return 0L;
}

void bufferSwitch(long index, ASIOBool processNow)
{
    // the actual processing callback.
    // Beware that this is normally in a seperate thread, hence be sure that you take care
    // about thread synchronization. This is omitted here for simplicity.

    // as this is a "back door" into the bufferSwitchTimeInfo a timeInfo needs to be created
    // though it will only set the timeInfo.samplePosition and timeInfo.systemTime fields and the according flags
    ASIOTime  timeInfo;
    memset (&timeInfo, 0, sizeof (timeInfo));

    // get the time stamp of the buffer, not necessary if no
    // synchronization to other media is required
    if(ASIOGetSamplePosition(&timeInfo.timeInfo.samplePosition, &timeInfo.timeInfo.systemTime) == ASE_OK)
        timeInfo.timeInfo.flags = kSystemTimeValid | kSamplePositionValid;

    bufferSwitchTimeInfo(&timeInfo, index, processNow);
}

long asioMessage(long selector, long value, void* message, double* opt)
{
    // currently the parameters "value", "message" and "opt" are not used.
    long ret = 0;
    switch(selector)
    {
    case kAsioSelectorSupported:
        if(value == kAsioResetRequest
            || value == kAsioEngineVersion
            || value == kAsioResyncRequest
            || value == kAsioLatenciesChanged
            // the following three were added for ASIO 2.0, you don't necessarily have to support them
            || value == kAsioSupportsTimeInfo
            || value == kAsioSupportsTimeCode
            || value == kAsioSupportsInputMonitor)
            ret = 1L;
        break;
    case kAsioResetRequest:
        // defer the task and perform the reset of the driver during the next "safe" situation
        // You cannot reset the driver right now, as this code is called from the driver.
        // Reset the driver is done by completely destruct is. I.e. ASIOStop(), ASIODisposeBuffers(), Destruction
        // Afterwards you initialize the driver again.
        ret = 1L;
        break;
    case kAsioResyncRequest:
        // This informs the application, that the driver encountered some non fatal data loss.
        // It is used for synchronization purposes of different media.
        // Added mainly to work around the Win16Mutex problems in Windows 95/98 with the
        // Windows Multimedia system, which could loose data because the Mutex was hold too long
        // by another thread.
        // However a driver can issue it in other situations, too.
        ret = 1L;
        break;
    case kAsioLatenciesChanged:
        // This will inform the host application that the drivers were latencies changed.
        // Beware, it this does not mean that the buffer sizes have changed!
        // You might need to update internal delay data.
        ret = 1L;
        break;
    case kAsioEngineVersion:
        // return the supported ASIO version of the host application
        // If a host applications does not implement this selector, ASIO 1.0 is assumed
        // by the driver
        ret = 2L;
        break;
    case kAsioSupportsTimeInfo:
        // informs the driver wether the asioCallbacks.bufferSwitchTimeInfo() callback
        // is supported.
        // For compatibility with ASIO 1.0 drivers the host application should always support
        // the "old" bufferSwitch method, too.
        ret = 1;
        break;
    case kAsioSupportsTimeCode:
        // informs the driver wether application is interested in time code info.
        // If an application does not need to know about time code, the driver has less work
        // to do.
        ret = 0;
        break;
    }
    return ret;
}

// Main:
int main()
{
    int retval = 0;
    bool inited = false, buffersCreated = false, started = false;
    char *error = NULL;

    if (!loadAsioDriver("UA-1000"))
    {
        error = "load failed.";
        goto err;
    }

    if (ASIOInit(&drv.driver) != ASE_OK)
        goto err;

    inited = true;

    if (ASIOGetChannels(&drv.inputChannels, &drv.outputChannels) != ASE_OK)
        goto err;

    printf("in: %d, out %d\n", drv.inputChannels, drv.outputChannels);

    if (ASIOGetBufferSize(&drv.minSize, &drv.maxSize, &drv.preferredSize, &drv.granularity) != ASE_OK)
        goto err;

    printf("min buf size: %d, max buf size: %d\n", drv.minSize, drv.maxSize);

    if (ASIOGetSampleRate(&drv.sampleRate) != ASE_OK)
        goto err;

    printf("rate: %f\n", drv.sampleRate);

    if (ASIOOutputReady() == ASE_OK)
        drv.postOutput = true;
    else
        drv.postOutput = false;

    // fill the bufferInfos from the start without a gap
    ASIOBufferInfo *info = drv.bufferInfos;

    // prepare inputs (Though this is not necessaily required, no opened inputs will work, too
    if (drv.inputChannels > kMaxInputChannels)
        drv.inputBuffers = kMaxInputChannels;
    else
        drv.inputBuffers = drv.inputChannels;
    for (int i = 0; i < drv.inputBuffers; i++, info++)
    {
        info->isInput = ASIOTrue;
        info->channelNum = i;
        info->buffers[0] = info->buffers[1] = 0;
    }

    // prepare outputs
    if (drv.outputChannels > kMaxOutputChannels)
        drv.outputBuffers = kMaxOutputChannels;
    else
        drv.outputBuffers = drv.outputChannels;
    for (int i = 0; i < drv.outputBuffers; i++, info++)
    {
        info->isInput = ASIOFalse;
        info->channelNum = i;
        info->buffers[0] = info->buffers[1] = 0;
    }

    asioCallbacks.asioMessage = asioMessage;
    asioCallbacks.bufferSwitch = bufferSwitch;
    asioCallbacks.bufferSwitchTimeInfo = bufferSwitchTimeInfo;

    // Create the buffers:
    if (ASIOCreateBuffers(drv.bufferInfos, drv.inputBuffers + drv.outputBuffers, drv.preferredSize, &asioCallbacks) != ASE_OK)
        goto err;
    else
        buffersCreated = true;

    // now get all the buffer details, sample word length, name, word clock group and activation
    for (int i = 0; i < drv.inputBuffers + drv.outputBuffers; i++)
    {
        drv.channelInfos[i].channel = drv.bufferInfos[i].channelNum;
        drv.channelInfos[i].isInput = drv.bufferInfos[i].isInput;
        if (ASIOGetChannelInfo(&drv.channelInfos[i]) != ASE_OK)
            goto err;

        printf("%s[%2d].type = %d\n", drv.channelInfos[i].isInput ? "in " : "out", drv.channelInfos[i].channel, drv.channelInfos[i].type);
        if (drv.channelInfos[i].type != ASIOSTInt32LSB)
        {
            error = "Application assumes sample types of ASIOSTInt32LSB!";
            goto err;
        }
    }

    // get the input and output latencies
    // Latencies often are only valid after ASIOCreateBuffers()
    // (input latency is the age of the first sample in the currently returned audio block)
    // (output latency is the time the first sample in the currently returned audio block requires to get to the output)
    if (ASIOGetLatencies(&drv.inputLatency, &drv.outputLatency) != ASE_OK)
        goto err;

    printf ("latencies: input: %d, output: %d\n", drv.inputLatency, drv.outputLatency);

    //goto done;

    // Start the engine:
    if (ASIOStart() != ASE_OK)
        goto err;
    else
        started = true;

    printf("Engine started.\n");
    for (int i = 0; i < 10; ++i)
    {
        Sleep(1000);

        printf("Engine running %2d.   \r", i);
    }

    goto done;

err:
    if (error == NULL)
        error = drv.driver.errorMessage;

    if (error != NULL)
        fprintf(stderr, "%s\r\n", error);

    retval = -1;

done:
    if (started)
        ASIOStop();
    if (buffersCreated)
        ASIODisposeBuffers();
    if (inited)
        ASIOExit();
    return retval;
}
