import sys, os, traceback
import math, io
import itertools, getopt
import json, zlib, random
import urllib.request
# pip install opencv-contrib-python
import numpy, cv2


# Result from yielded frame
def inference( frame_generator ):

    # Guess copyright
    grouped, results = [], []
    previous = None
    for framecounter, (frame, offset) in enumerate(frame_generator):
        assert type(frame) is numpy.ndarray and frame.shape[2] == 3
        # ask
        lookup = postframe(frame)
        if lookup is None:
            continue
        # yield server-result
        if previous:
            yield *previous, None
        previous = frame, offset, lookup
        # ignore if frame too common
        if len(lookup) >= 20:
            continue
        # whitelist result if frame also in trailer/teaser/...
        lookup_whitelist = [result for result in lookup if not any(True for f in result['frames'] if f['type'] == 'trailer')]
        if not lookup_whitelist:
            continue
        # group by uri (3 matches -> unlikely false positive)
        results.append(lookup_whitelist)
        grouped = [list(group) for k, group in
                   itertools.groupby(sorted([item for result in results for item in result], key=lambda x: x['uri']), lambda x: x['uri'])]
        grouped.sort(key=lambda x:len(x), reverse=True)
        if len(set(frame['offset'] for result in grouped[0] for frame in result['frames'] if frame['matrix'] is not None)) >= 3:
            break
    if not previous:
        return

    # To filter, or not to filter: that is the question...
    copyrights = []
    for group in grouped:
        copyright = False
        # video: accurate if 3 different frame-offset
        if len(set(frame['offset'] for result in group for frame in result['frames'] if frame['matrix'] is not None)) >= 3:
            copyright = True
        # still-image or short-video: if matches 'image' or translation-matrix + perceptual-hash
        elif framecounter <= 2 and any(True for result in group for frame in result['frames'] if
                frame['type'] == 'image' or (frame['matrix'] is not None and frame['hamming'] is not None)):
            copyright = True
        if copyright:
            copyrights.append(group[0])
    yield *previous, copyrights or None


# ask server
def postframe( frame ):

    # filepath or numpy.array
    assert type(frame) is numpy.ndarray
    res, pngimage = cv2.imencode('.png', frame)
    content_type = 'image/png'
    filename = 'frame.png'
    filedata = pngimage.tobytes()

    # https://bugs.python.org/issue3244
    url = 'https://framespot.com/'
    boundary = ''.join(random.choices('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz', k=70)) # RFC2046: boundary must be no longer than 70 characters
    headers = {
        'Content-Type': 'multipart/form-data; boundary=%s' % boundary,
        'Accept': 'application/json',
        'Accept-Encoding': 'gzip, deflate',
        'User-Agent': 'framespot-client/1.0',
    }
    data = (b'--%s\r\n' % boundary.encode() +
            b'Content-Disposition: form-data; name="frame"; filename="%s"\r\n' % filename.encode() +
            b'Content-Type: %s\r\n\r\n' % content_type.encode() +
            filedata + b'\r\n'
            b'--%s--\r\n' % boundary.encode())
    try:
        request = urllib.request.Request(url, method='POST', headers=headers, data=data)
        with urllib.request.urlopen(request, timeout=120) as response:
            result_code = response.getcode()
            result_url = response.geturl()
            result_headers = response.info()
            result_type = result_headers.get_content_type()
            if result_code != 200:
                return None
            assert result_url == url and result_type == 'application/json'

            # Uncompress
            decompressor = None
            if result_headers.get('Content-Encoding') == 'zlib':
                decompressor = zlib.decompressobj()
            elif result_headers.get('Content-Encoding') == 'gzip':
                decompressor = zlib.decompressobj(zlib.MAX_WBITS|16)
            elif result_headers.get('Content-Encoding') == 'deflate':
                decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
            elif result_headers.get('Content-Encoding'):
                decompressor = zlib.decompressobj(zlib.MAX_WBITS|32) # automatic header detection
            result_data = b''
            while True:
                buf = response.read(0x1000)
                if not buf:
                    break
                result_data += decompressor.decompress(buf) if decompressor else buf
                assert len(result_data) < 0x1000000
            if decompressor:
                result_data += decompressor.flush()
        return json.loads(result_data)

    except (urllib.error.HTTPError, urllib.error.URLError):
        traceback.print_exc(file=sys.stderr)
    return None


# scenecut @ 500 fps
def scenecut(filepath, scene_min=None, scene_max=None, seek=None, duration=None):

    kp_detector = cv2.FastFeatureDetector_create()
    kp_descriptor = cv2.ORB_create()
    bf_hamming = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    pyramid_down = 240
    truncate_keypoints = 256
    min_kpmatched = 8
    bits_kpmatched = 32
    min_keypoints = 256
    # Brief is faster than ORB - but not rotation invariant
    if hasattr(cv2, 'xfeatures2d'):
        kp_descriptor = cv2.xfeatures2d.BriefDescriptorExtractor_create(bytes=16)
        bits_kpmatched = 16

    cap = cv2.VideoCapture(filepath)
    if seek is not None:
        cap.set(cv2.CAP_PROP_POS_MSEC, seek)

    best_frame = None
    best_offset = None
    best_quality = 0.0
    des_prev = None
    scene_start = (0 if seek is None else seek)
    stop = scene_start + duration if duration else None

    while True:
        ret, frame = cap.read()
        cap_time = cap.get(cv2.CAP_PROP_POS_MSEC)
        if not ret or (best_frame is not None and stop is not None and cap_time >= stop):
            if best_frame is not None:
                yield (best_frame, best_offset)
            break

        # Keypoints on simplified frame
        grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        width = math.ceil(math.sqrt(pyramid_down*pyramid_down * grayframe.shape[1] / grayframe.shape[0]))
        height = math.ceil(math.sqrt(pyramid_down*pyramid_down * grayframe.shape[0] / grayframe.shape[1]))
        smallframe = cv2.resize(grayframe, (width, height), interpolation=cv2.INTER_AREA)

        kpf = kp_detector.detect(smallframe, None)
        kpf = list(kpf) # opencv 4.5.4 replaced list-results with tuples
        if len(kpf) < min_keypoints:
            continue
        kpf.sort(key=lambda kp: kp.response, reverse=True)
        kps, des = kp_descriptor.compute(smallframe, kpf[:truncate_keypoints])

        # Best frame within scene
        if des_prev is not None and des is not None:

            # Scene cut?
            newscene = False
            if scene_max is not None and scene_start + scene_max < cap_time:
                newscene = True
            elif scene_min is None or scene_start + scene_min < cap_time:
                matches = bf_hamming.match(des_prev, des)
                matched = list(filter(lambda match: match.distance <= bits_kpmatched, matches))
                if len(matched) < min_kpmatched:
                    newscene = True

            # Yield frame
            if newscene and best_frame is not None:
                yield (best_frame, best_offset)
                scene_start = cap_time
                best_frame = None
                best_offset = None
                best_quality = 0.0

            # Better frame?
            else:
                quality = sum(kp.response for kp in kps)
                if best_quality < quality:
                    best_frame = frame
                    best_offset = cap_time
                    best_quality = quality
        des_prev = des

    cap.release()


# main
if __name__ == '__main__':

    # Params
    opts, args = getopt.getopt(sys.argv[1:],'s:d:v',['seek=','duration=','min-scene=','max-scene=','verbose'])
    if len(args) != 1:
        print('Usage: python3 . --seek=#sec --duration=#sec /path/to/file', file=sys.stderr)
        sys.exit(os.EX_USAGE)
    filepath = args[0]
    if not os.path.exists(filepath):
        print('File not found:', filepath, file=sys.stderr)
        sys.exit(os.EX_USAGE)
    scene_min, scene_max = 5000, 60000  # scene: [5s..60s]
    seek, duration = None, None
    verbose = False
    for o, a in opts:
        if o in ('-s', '--seek'):
            seek = float(a) * 1000
        elif o in ('-d', '--duration'):
            duration = float(a) * 1000
        elif o in ('min-scene'):
            scene_min = float(a) * 1000
        elif o in ('max-scene'):
            scene_max = float(a) * 1000
        elif o in ('-v','--verbose'):
            verbose = True

    if verbose:
        print('Inference:', filepath, 'seek:',seek, 'duration:',duration, 'scene:['+str(scene_min)+':'+str(scene_max)+']', file=sys.stderr)

    # Detect video (container only, could also match an audio)
    is_video = False
    with open(filepath, 'rb') as fp:
        buf = bytearray(fp.read(8192))
    # video/mp4 (.mp4) + video/quicktime (.mov) + video/x-m4v (.m4v)
    if len(buf) > 8 and buf[4] == 0x66 and buf[5] == 0x74 and buf[6] == 0x79 and buf[7] == 0x70:
        ftyp_len = int.from_bytes(buf[0:4], byteorder='big')
        if len(buf) > 10 and buf[0] == 0x0 and buf[1] == 0x0 and buf[2] == 0x0 and buf[3] == 0x1C and buf[8] == 0x4D and buf[9] == 0x34 and buf[10] == 0x56:
            is_video = True
        elif len(buf) >= ftyp_len:
            major_brand = buf[8:12].decode(errors='ignore')
            compatible_brands = [buf[i:i+4].decode(errors='ignore') for i in range(16, ftyp_len, 4)]
            if major_brand in ['mp41','mp42','isom','qt  ']:
                is_video = True
            elif 'mp41' in compatible_brands or 'mp42' in compatible_brands or 'isom' in compatible_brands:
                is_video = True
    # video/webm (.webm) + video/x-matroska (.mkv)
    elif buf.startswith(b'\x1A\x45\xDF\xA3') and (buf.find(b'\x42\x82\x84webm') > -1 or buf.find(b'\x42\x82\x88matroska') > -1):
        is_video = True
    # video/mpeg (.mpg)
    elif len(buf) > 3 and buf[0] == 0x0 and buf[1] == 0x0 and buf[2] == 0x1 and buf[3] >= 0xb0 and buf[3] <= 0xbf:
        is_video = True
    # video/mp2t (.ts)
    #elif len(buf) > 12 and buf[0] == 0x47 and ...:
    #    is_video = True
    # video/x-msvideo (.avi)
    elif len(buf) > 11 and buf[0] == 0x52 and buf[1] == 0x49 and buf[2] == 0x46 and buf[3] == 0x46 and buf[8] == 0x41 and buf[9] == 0x56 and buf[10] == 0x49 and buf[11] == 0x20:
        is_video = True
    # video/x-ms-wmv (.wmv)
    elif len(buf) > 9 and buf[0] == 0x30 and buf[1] == 0x26 and buf[2] == 0xB2 and buf[3] == 0x75 and buf[4] == 0x8E and buf[5] == 0x66 and buf[6] == 0xCF and buf[7] == 0x11 and buf[8] == 0xA6 and buf[9] == 0xD9:
        is_video = True
    # video/3gpp (.3gp)
    elif len(buf) > 7 and buf[0] == 0x66 and buf[1] == 0x74 and buf[2] == 0x79 and buf[3] == 0x70 and buf[4] == 0x33 and buf[5] == 0x67 and buf[6] == 0x70:
        is_video = True
    # video/x-flv (.flv)
    elif len(buf) > 3 and buf[0] == 0x46 and buf[1] == 0x4C and buf[2] == 0x56 and buf[3] == 0x01:
        is_video = True
    # image/gif (.gif)
    elif len(buf) > 2 and buf[0] == 0x47 and buf[1] == 0x49 and buf[2] == 0x46:
        if b'\x21\xFF\x0B\x4E\x45\x54\x53\x43\x41\x50\x45\x32\x2E\x30' in buf:  # animated
            is_video = True
    # image/webp (.webp)
    elif len(buf) > 16 and buf[0] == 0x52 and buf[1] == 0x49 and buf[2] == 0x46 and buf[3] == 0x46 and buf[8] == 0x57 and buf[9] == 0x45 and buf[10] == 0x42 and buf[11] == 0x50 and buf[12] == 0x56 and buf[13] == 0x50:
        if buf[12:16] == b'VP8X' and buf[16] & 2 != 0:  # animated
            is_video = True

    # Frame generator
    if is_video:
        frame_generator = scenecut(filepath, scene_min=scene_min, scene_max=scene_max, seek=seek, duration=duration)
    else:
        frame = cv2.imread( filepath, cv2.IMREAD_UNCHANGED )
        if frame is None or frame.dtype != numpy.uint8 or len(frame.shape) == 2 or frame.shape[2] != 3:
            try:
                import PIL.Image
                with PIL.Image.open(filepath) as img:
                    if img.mode != 'RGB':
                        white_background = PIL.Image.new('RGBA', img.size, (255,255,255))
                        img = PIL.Image.alpha_composite(white_background, img.convert('RGBA')).convert('RGB')
                    frame = numpy.array(img, dtype=numpy.uint8)[...,::-1].copy() # RGB->BGR
            except ImportError:
                frame = None
            except PIL.UnidentifiedImageError:
                frame = None
        if frame is None:
            print('Could not open', filepath, file=sys.stderr)
            sys.exit(os.EX_NOINPUT)
        frame_generator = [(frame, None)]

    # Lookup frames
    got_frames = False
    for frame, offset, lookup, copyrights in inference(frame_generator):
        got_frames = True
        if verbose:
            label = '{:02d}:{:02d}:{:02d}'.format(int(offset/3600000) % 24,int(offset/60000) % 60,int(offset/1000) % 60) if type(offset) in [float,int] else offset
            print(label, 'response:', json.dumps(lookup), file=sys.stderr)
            frameoffset = frame
            cv2.putText(frameoffset,label, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
            cv2.imshow('frame',frameoffset)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                quit()
        if copyrights:
            print(json.dumps(copyrights, indent=2))
            break

    if not got_frames:
        print('Did not yield frames', file=sys.stderr)
        sys.exit(os.EX_NOINPUT)
    if verbose:
        cv2.waitKey()

