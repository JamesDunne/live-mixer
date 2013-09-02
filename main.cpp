// Simple ASIO engine test application.
//#define NOT_LIVE

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <float.h>

#include <xmmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>

#include "asiosys.h"
#include "asio.h"
#include "asiodrivers.h"

#include "avx.h"

extern AsioDrivers* asioDrivers;
extern bool loadAsioDriver(char *name);

// Must be a multiple of 8:
const long inputChannels = 8L;
const long icr = inputChannels * sizeof(double) / sizeof(vec4_d64);

enum
{
    // number of input and outputs supported by the host application
    // you can change these to higher or lower values
    kMaxInputChannels = 8,
    kMaxOutputChannels = 8
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

void print_dB(double v)
{
    int fpc = _fpclass(v);
    if (fpc == _FPCLASS_NINF)
        printf("    -INF");
    else if (fpc == _FPCLASS_PINF)
        printf("    +INF");
    else
        printf("%8.2f", v);
}

void printvec_dB(vec4_d64 v)
{
    print_dB(v.m256d_f64[0]);
    printf(" ");
    print_dB(v.m256d_f64[1]);
    printf(" ");
    print_dB(v.m256d_f64[2]);
    printf(" ");
    print_dB(v.m256d_f64[3]);
}

void printvec_samp(vec8_i32 v)
{
    printf(
        //"%11d %11d %11d %11d %11d %11d %11d %11d",
        "%08x %08x %08x %08x %08x %08x %08x %08x",
        v.m256i_i32[0],
        v.m256i_i32[1],
        v.m256i_i32[2],
        v.m256i_i32[3],
        v.m256i_i32[4],
        v.m256i_i32[5],
        v.m256i_i32[6],
        v.m256i_i32[7]
    );
}

// State and monitoring levels for the entire effects chain:
typedef struct {
    // Reported input dBFS levels:
    struct {
        vec4_dBFS       levels[icr];
    } fi_monitor;

    // Input gain:
    struct {
        // Input values in dB:
        struct {
            vec4_dB     gain[icr];
        } input;

        // Calculated linear scalars:
        struct {
            vec4_scalar gain[icr];
        } calc;

        // Initialize all values:
        void init()
        {
            for (int i = 0; i < icr; ++i)
            {
                input.gain[i]   = _mm256_set1_pd(0.0);
                calc.gain[i]    = _mm256_set1_pd(1.0);
            }
        }

        // Recalculate input-dependent values:
        void recalc()
        {
            for (int i = 0; i < icr; ++i)
            {
                calc.gain[i] = dB_to_scalar(input.gain[i]);
            }
        }
    } f0_gain;

    // Reported post-gain dBFS levels:
    struct {
        vec4_dBFS       levels[icr];
    } f0_output;

    // Compressor:
    struct {
        struct {
            vec4_dBFS   threshold[icr];
            vec4_msec   attack[icr];
            vec4_msec   release[icr];
            vec4_dB     gain[icr];
            vec4_scalar ratio[icr];
        } input;

        struct {
            // (input.ratio - 1.0)
            vec4_scalar ratio_min_1[icr];

            // coef = exp( -1000.0 / ( ms * sampleRate ) )
            vec4_scalar attack_coef[icr];
            vec4_scalar release_coef[icr];

            // dB_to_scalar(gain)
            vec4_scalar gain[icr];
        } calc;

        // Working state:
        struct {
            vec4_dB     env[icr];
        } state;

        struct {
            vec4_dB     gain_reduction[icr];
        } monitor;

        // Initialize all values:
        void init()
        {
            for (int i = 0; i < icr; ++i)
            {
                input.threshold[i]  = _mm256_set1_pd(0);     // dBFS
                input.attack[i]     = _mm256_set1_pd(1);     // msec
                input.release[i]    = _mm256_set1_pd(200);   // msec
                input.ratio[i]      = _mm256_set1_pd(1);     // N:1
                input.gain[i]       = _mm256_set1_pd(0);     // dB

                state.env[i] = DC_OFFSET;
                monitor.gain_reduction[i] = _mm256_set1_pd(0);
            }
        }

        // Recalculate input-dependent values:
        void recalc()
        {
            for (int i = 0; i < icr; ++i)
            {
                calc.ratio_min_1[i]     = _mm256_sub_pd(input.ratio[i], _mm256_set1_pd(1.0));
                calc.attack_coef[i]     = mm256_exp_pd( _mm256_div_pd( _mm256_set1_pd(-1000.0), _mm256_mul_pd(input.attack[i], _mm256_set1_pd(drv.sampleRate)) ) );
                calc.release_coef[i]    = mm256_exp_pd( _mm256_div_pd( _mm256_set1_pd(-1000.0), _mm256_mul_pd(input.release[i], _mm256_set1_pd(drv.sampleRate)) ) );
                calc.gain[i]            = dB_to_scalar(input.gain[i]);
            }
        }
    } f1_compressor;

    // Reported output dBFS levels:
    struct {
        vec4_dBFS       levels[icr];
    } fo_monitor;
} EffectParameters;

EffectParameters fx;

// Process audio effects for 8 channels simultaneously:
void processEffects(const vec8_i32 &inpSamples, vec8_i32 &outSamples, const long n)
{
    // Extract int samples and convert to doubles:
    const vec4_d64 ds0 = _mm256_div_pd(
        _mm256_cvtepi32_pd(_mm256_extractf128_si256(inpSamples, 0)),
        _mm256_set1_pd((double)INT_MAX)
        );
    const vec4_d64 ds1 = _mm256_div_pd(
        _mm256_cvtepi32_pd(_mm256_extractf128_si256(inpSamples, 1)),
        _mm256_set1_pd((double)INT_MAX)
        );

    // Monitor input levels:
    fx.fi_monitor.levels[n + 0] = scalar_to_dBFS(ds0);
    fx.fi_monitor.levels[n + 1] = scalar_to_dBFS(ds1);

    vec4_d64 s0, s1;

    // f0_gain:
    {
        s0 = _mm256_mul_pd(ds0, fx.f0_gain.calc.gain[n + 0]);
        s1 = _mm256_mul_pd(ds1, fx.f0_gain.calc.gain[n + 1]);
    }

    // Monitor levels:
    fx.f0_output.levels[n + 0] = scalar_to_dBFS(s0);
    fx.f0_output.levels[n + 1] = scalar_to_dBFS(s1);

    // f1_compressor:
    {
        const vec4_dBFS l0 = scalar_to_dBFS_offs(s0);
        const vec4_dBFS l1 = scalar_to_dBFS_offs(s1);

        // over = s - thresh
        vec4_dB over0 = _mm256_sub_pd(l0, fx.f1_compressor.input.threshold[n + 0]);
        vec4_dB over1 = _mm256_sub_pd(l1, fx.f1_compressor.input.threshold[n + 1]);

        // over = if over < 0.0 then 0.0 else over;
        over0 = mm256_if_then_else(_mm256_cmp_pd(over0, _mm256_set1_pd(0.0), _CMP_LT_OQ), _mm256_set1_pd(0.0), over0);
        over1 = mm256_if_then_else(_mm256_cmp_pd(over1, _mm256_set1_pd(0.0), _CMP_LT_OQ), _mm256_set1_pd(0.0), over1);

        // over += DC_OFFSET
        over0 = _mm256_add_pd(over0, DC_OFFSET);
        over1 = _mm256_add_pd(over1, DC_OFFSET);

        // env = over + coef * ( env - over )
        const vec4_dB attack_env0  = _mm256_add_pd(over0, _mm256_mul_pd(fx.f1_compressor.calc.attack_coef[n + 0], _mm256_sub_pd(fx.f1_compressor.state.env[n + 0], over0)));
        const vec4_dB attack_env1  = _mm256_add_pd(over1, _mm256_mul_pd(fx.f1_compressor.calc.attack_coef[n + 1], _mm256_sub_pd(fx.f1_compressor.state.env[n + 1], over1)));
        const vec4_dB release_env0  = _mm256_add_pd(over0, _mm256_mul_pd(fx.f1_compressor.calc.release_coef[n + 0], _mm256_sub_pd(fx.f1_compressor.state.env[n + 0], over0)));
        const vec4_dB release_env1  = _mm256_add_pd(over1, _mm256_mul_pd(fx.f1_compressor.calc.release_coef[n + 1], _mm256_sub_pd(fx.f1_compressor.state.env[n + 1], over1)));

        // env = if over > env then attack_env else release_env
        fx.f1_compressor.state.env[n + 0] = mm256_if_then_else(_mm256_cmp_pd(over0, fx.f1_compressor.state.env[n + 0], _CMP_GT_OQ), attack_env0, release_env0);
        fx.f1_compressor.state.env[n + 1] = mm256_if_then_else(_mm256_cmp_pd(over1, fx.f1_compressor.state.env[n + 1], _CMP_GT_OQ), attack_env1, release_env1);

        // over = env - DC_OFFSET
        over0 = _mm256_sub_pd(fx.f1_compressor.state.env[n + 0], DC_OFFSET);
        over1 = _mm256_sub_pd(fx.f1_compressor.state.env[n + 1], DC_OFFSET);

        // grdB = ( over * ( ratio - 1.0 ) )
        vec4_dB gr0dB = _mm256_mul_pd(over0, fx.f1_compressor.calc.ratio_min_1[n + 0]);
        vec4_dB gr1dB = _mm256_mul_pd(over1, fx.f1_compressor.calc.ratio_min_1[n + 1]);

        // gr = dB_to_scalar(grdB)
        fx.f1_compressor.monitor.gain_reduction[n + 0] = dB_to_scalar(gr0dB);
        fx.f1_compressor.monitor.gain_reduction[n + 1] = dB_to_scalar(gr1dB);

        // Apply gain reduction to inputs:
        s0 = _mm256_mul_pd(s0, fx.f1_compressor.monitor.gain_reduction[n + 0]);
        s1 = _mm256_mul_pd(s1, fx.f1_compressor.monitor.gain_reduction[n + 1]);

        // Apply make-up gain:
        s0 = _mm256_mul_pd(s0, fx.f1_compressor.calc.gain[n + 0]);
        s1 = _mm256_mul_pd(s1, fx.f1_compressor.calc.gain[n + 1]);
    }

    // Monitor output levels:
    fx.fo_monitor.levels[n + 0] = scalar_to_dBFS(s0);
    fx.fo_monitor.levels[n + 1] = scalar_to_dBFS(s1);

    // TODO(jsd): Better limiter implementation!
    // Limit final samples:
    s0 = _mm256_max_pd(_mm256_min_pd(s0, _mm256_set1_pd((double)1.0)), _mm256_set1_pd((double)-1.0));
    s1 = _mm256_max_pd(_mm256_min_pd(s1, _mm256_set1_pd((double)1.0)), _mm256_set1_pd((double)-1.0));

    // Convert doubles back to 32-bit ints:
    s0 = _mm256_mul_pd(s0, _mm256_set1_pd((double)INT_MAX));
    s1 = _mm256_mul_pd(s1, _mm256_set1_pd((double)INT_MAX));
    const vec8_i32 os = _mm256_setr_m128i(_mm256_cvtpd_epi32(s0), _mm256_cvtpd_epi32(s1));

    // Write outputs:
    _mm256_stream_si256(&outSamples, os);
}

// Main audio processing callback.
// NOTE: Called on a separate thread from main() thread.
ASIOTime *bufferSwitchTimeInfo(ASIOTime *timeInfo, long index, ASIOBool processNow)
{
    // Buffer size (in samples):
    long buffSize = drv.preferredSize;

    // Assume the buffer size is an even multiple of 32-bytes:
    assert((buffSize % sizeof(vec4_d64)) == 0);

    for (long i = 0; i < buffSize; ++i)
    {
        assert(index == 0 || index == 1);

        // Process 8 channels of 32-bit samples per iteration:
        for (long n = 0; n < inputChannels / 8; ++n)
        {
            const long ci = n * 8;
            const long co = drv.inputBuffers + ci;

            // Stripe input samples into a vector:
            const vec8_i32 inpSamples = _mm256_setr_epi32(
                ((long *)drv.bufferInfos[ci + 0].buffers[index])[i],
                ((long *)drv.bufferInfos[ci + 1].buffers[index])[i],
                ((long *)drv.bufferInfos[ci + 2].buffers[index])[i],
                ((long *)drv.bufferInfos[ci + 3].buffers[index])[i],
                ((long *)drv.bufferInfos[ci + 4].buffers[index])[i],
                ((long *)drv.bufferInfos[ci + 5].buffers[index])[i],
                ((long *)drv.bufferInfos[ci + 6].buffers[index])[i],
                ((long *)drv.bufferInfos[ci + 7].buffers[index])[i]
            );

            // Process audio effects:
            vec8_i32 outSamples;
            processEffects(inpSamples, outSamples, n * 2);
            // Copy outputs to output channel buffers:
            const long *outputs32 = (const long *)&outSamples;
            ((long *)drv.bufferInfos[co + 0].buffers[index])[i] = outputs32[0];
            ((long *)drv.bufferInfos[co + 1].buffers[index])[i] = outputs32[1];
            ((long *)drv.bufferInfos[co + 2].buffers[index])[i] = outputs32[2];
            ((long *)drv.bufferInfos[co + 3].buffers[index])[i] = outputs32[3];
            ((long *)drv.bufferInfos[co + 4].buffers[index])[i] = outputs32[4];
            ((long *)drv.bufferInfos[co + 5].buffers[index])[i] = outputs32[5];
            ((long *)drv.bufferInfos[co + 6].buffers[index])[i] = outputs32[6];
            ((long *)drv.bufferInfos[co + 7].buffers[index])[i] = outputs32[7];
        }
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

    drv.sampleRate = 44100.0;

    // Initialize FX parameters:
    fx.f0_gain.init();
    fx.f1_compressor.init();

    // Set our own inputs:
    for (int i = 0; i < icr; ++i)
    {
        fx.f0_gain.input.gain[i] = _mm256_set1_pd(0);   // dB

        fx.f1_compressor.input.threshold[i] = _mm256_set1_pd(-30);  // dBFS
        fx.f1_compressor.input.attack[i]    = _mm256_set1_pd(1.0);  // msec
        fx.f1_compressor.input.release[i]   = _mm256_set1_pd(80);   // msec
        fx.f1_compressor.input.ratio[i]     = _mm256_set1_pd(0.25); // N:1
        fx.f1_compressor.input.gain[i]      = _mm256_set1_pd(6);    // dB
    }

    // Calculate input-dependent values:
    fx.f0_gain.recalc();
    fx.f1_compressor.recalc();

    // FX parameters are all set.

#ifdef NOT_LIVE
    // Test mode:

#if 0
    const auto t0 = mm256_if_then_else(_mm256_cmp_pd(_mm256_set1_pd(-1.0), _mm256_set1_pd(0.0), _CMP_LT_OQ), _mm256_set1_pd(0.0), _mm256_set1_pd(-1.0));
    printvec_dB(t0);
    printf("\n\n");
    const auto p0 = mm256_if_then_else(_mm256_cmp_pd(_mm256_set1_pd(-1.0), _mm256_set1_pd(0.0), _CMP_LT_OQ), _mm256_set1_pd(0.0), _mm256_set1_pd(1.0));
    printvec_dB(t0);
    printf("\n\n");
    const auto t1 = mm256_if_then_else(_mm256_cmp_pd(_mm256_set1_pd(0.0), _mm256_set1_pd(0.0), _CMP_LT_OQ), _mm256_set1_pd(0.0), _mm256_set1_pd(-1.0));
    printvec_dB(t1);
    printf("\n\n");
    const auto p1 = mm256_if_then_else(_mm256_cmp_pd(_mm256_set1_pd(0.0), _mm256_set1_pd(0.0), _CMP_LT_OQ), _mm256_set1_pd(0.0), _mm256_set1_pd(1.0));
    printvec_dB(t1);
    printf("\n\n");
    goto done;
#endif

    vec8_i32 in, out;
    long long c = 0LL;
    for (int i = 0; i < 20; ++i)
    {
        for (int n = 0; n < 48; ++n, ++c)
        {
            double s = sin(2.0 * 3.14159265358979323846 * (double)c / drv.sampleRate);
            int si = (int)(s * INT_MAX / 2);

            in = _mm256_set1_epi32(si);

            processEffects(in, out, 0);
        }

#if 1
        printf("samp:   ");
        printvec_samp(in);
        printf("\n");

        printf("input:  ");
        for (int n = 0; n < icr; ++n)
        {
            printvec_dB(fx.fi_monitor.levels[n]);
            if (n < icr - 1) printf(" ");
        }
        printf("\n");

        printf("gain:   ");
        for (int n = 0; n < icr; ++n)
        {
            printvec_dB(fx.f0_output.levels[n]);
            if (n < icr - 1) printf(" ");
        }
        printf("\n");

        printf("comp:   ");
        for (int n = 0; n < icr; ++n)
        {
            printvec_dB(fx.fo_monitor.levels[n]);
            if (n < icr - 1) printf(" ");
        }
        printf("\n");

        printf("samp:   ");
        printvec_samp(out);
        printf("\n\n");
#endif
    }
#else
    // ASIO live engine mode:
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

    printf("min buf size: %d, preferred: %d, max buf size: %d\n", drv.minSize, drv.preferredSize, drv.maxSize);

    if (ASIOGetSampleRate(&drv.sampleRate) != ASE_OK)
        goto err;

    printf("rate: %f\n\n", drv.sampleRate);

    if (ASIOOutputReady() == ASE_OK)
        drv.postOutput = true;
    else
        drv.postOutput = false;

    // fill the bufferInfos from the start without a gap
    ASIOBufferInfo *info = drv.bufferInfos;

    // prepare inputs (Though this is not necessarily required, no opened inputs will work, too
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

        //printf("%s[%2d].type = %d\n", drv.channelInfos[i].isInput ? "in " : "out", drv.channelInfos[i].channel, drv.channelInfos[i].type);
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

    // Start the engine:
    if (ASIOStart() != ASE_OK)
        goto err;
    else
        started = true;

    printf("Engine started.\n\n");
    const int total_time = 30;
    for (int i = 0; i < total_time; ++i)
    {
        printf("Engine running %2d.   \r", total_time - i);
        Sleep(1000);
    }
#endif

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
