import sys, os, traceback
import math, io
import itertools, getopt
import json, zlib, random
import imghdr, urllib.request
# pip install opencv-contrib-python
import numpy, cv2


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

    except urllib.error.HTTPError:
        traceback.print_exc(file=sys.stderr)
    return None


# yield frame + result
def inference( frame_generator ):

    # Guess copyright
    framecounter = 0
    grouped, results = [], []
    previous = None
    for frame, offset in frame_generator:
        framecounter += 1
        # filepath?
        if type(frame) is str:
            try:
                import PIL.Image
                with PIL.Image.open(frame) as img:
                    white_background = PIL.Image.new('RGBA', img.size, (255,255,255))
                    img = PIL.Image.alpha_composite(white_background, img.convert('RGBA')).convert('RGB')
                    frame = numpy.array(img, dtype=numpy.uint8)[...,::-1].copy() # RGB->BGR
            except ImportError:
                frame = cv2.imread(frame, cv2.IMREAD_COLOR)
            except PIL.UnidentifiedImageError:
                continue
        # ask
        lookup = postframe(frame)
        if lookup is None:
            continue
        # yield server-result
        if previous:
            yield *previous, None
        previous = frame, offset, lookup
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
    # Ignore if frame too common
    if len(copyrights) >= 20:
        copyrights = None
    yield *previous, copyrights or None


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


# Monkeypatch imghdr - https://bugs.python.org/issue28591
def imghdr_jpeg1(h, f):
    if b'JFIF' in h[:23]:
        return 'jpeg'
def imghdr_jpeg2(h, f):
    if len(h) >= 32 and 67 == h[5] and h[:32] == b'\xff\xd8\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f':
        return 'jpeg'
def imghdr_jpeg3(h, f):
    if h[6:10] in (b'JFIF', b'Exif') or h[:2] == b'\xff\xd8':
        return 'jpeg'
imghdr.tests.append(imghdr_jpeg1)
imghdr.tests.append(imghdr_jpeg2)
imghdr.tests.append(imghdr_jpeg3)


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
    frame_generator = []
    if imghdr.what(filepath):
        offset = os.path.basename(filepath)
        frame_generator.append((filepath, offset))
    else:
        frame_generator = scenecut(filepath, scene_min=scene_min, scene_max=scene_max, seek=seek, duration=duration)

    # Lookup frames
    frame_counter = 0
    for frame, offset, lookup, copyrights in inference(frame_generator):
        frame_counter += 1
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

    if frame_counter == 0:
        print('Did not yield frames', file=sys.stderr)
        sys.exit(os.EX_NOINPUT)
    if verbose:
        cv2.waitKey()

