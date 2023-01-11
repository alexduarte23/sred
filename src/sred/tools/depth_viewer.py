import cv2
import numpy as np
import pathlib
import PySimpleGUI as sg
import argparse
import sys

if sys.path[-1] != str(pathlib.Path(__file__).parents[1]):
    sys.path.append(str(pathlib.Path(__file__).parents[1]))

import sred



def _sanitize_input_val(input_str, min_val, max_val):
    ''' Sanitize and convert str into valid number in range '''

    if input_str == '' or input_str == None:
        return max(min(0, max_val), min_val)

    sign = -1 if input_str[0] == '-' else 1

    whole = input_str.split('.', 1)[0]
    digits = ''.join(c for c in whole if c.isdigit())
    if digits == '':
        return max(min(0, max_val), min_val)

    num = sign * int(digits)
    return  max(min(num, max_val), min_val)


def _update_frame_info(window, image, zoom, center, val_range):
    ''' Update UI text info with current frame stats '''

    h, w = image.shape
    h_new = h / zoom
    w_new = w / zoom
    h_min = center[1] - h_new/2
    h_min = min(max(h_min, 0), h-h_new)
    h_max = h_min + h_new
    w_min = center[0] - w_new/2
    w_min = min(max(w_min, 0), w-w_new)
    w_max = w_min + w_new
    image = image[round(h_min):round(h_max), round(w_min):round(w_max)]
    
    stats = sred.utils.get_stats(image, val_range)

    window['-INFO MIN-'].update(value=f'Min value: {stats["range"][0]}')
    window['-INFO MAX-'].update(value=f'Max value: {stats["range"][1]}')
    window['-INFO HOLES-'].update(value=f'Holes: {stats["holes"]} ({stats["holes"]/stats["n"]*100:.1f}%)')
    window['-INFO CLIP-'].update(value=f'Clipped points: min {stats["clipped"][0]} ({stats["clipped"][0]/stats["n"]*100:.1f}%) | max {stats["clipped"][1]} ({stats["clipped"][1]/stats["n"]*100:.1f}%)')


def _update_mouse_hover_info(window, fitted, mouse_pos, val_range):
    ''' Update UI text info with current mouse position over image '''

    if mouse_pos == None:
        info = '---'
    else:
        value = fitted[mouse_pos[1], mouse_pos[0]]
        if value == 0: info = f'{value} (hole)'
        elif value > val_range[1]: info = f'{value} (clipped to {val_range[1]})'
        elif value < val_range[0]: info = f'{value} (clipped to {val_range[0]})'
        else: info = f'{value}'

    window['-INFO CURSOR-'].update(value='Cursor: ' + info)


def _create_viewer_window(img_size, num_ports, type_limits, frame_count):
    ''' Create GUI window and layout '''

    img_w, img_h = img_size
    type_min, type_max = type_limits

    sg.theme('SystemDefault')

    controls_col = [
        #[sg.Button('-1', key='-PREV-'), sg.Button('Play', key='-PLAY-'), sg.Button('+1', key='-NEXT-')],
        [sg.Checkbox('Mark clipping (blue, red)', default=True, key='-CLIP-')],
        [sg.Checkbox('Mark holes (pink)', key='-HOLES-')],
        [sg.Text("Min", justification="left")],
        [sg.Slider(range=(type_min,type_max), size=(30,12), default_value=type_min, resolution=1, orientation='h', disable_number_display=True, enable_events=True, key='-MIN-'), 
            sg.Input(str(type_min), size=(7,None), key='-MININ-')],
        [sg.Text("Max", justification="left")],
        [sg.Slider(range=(type_min,type_max), size=(30,12), default_value=type_max, resolution=1, orientation='h', disable_number_display=True, enable_events=True, key='-MAX-'), 
            sg.Input(str(type_max), size=(7,None), key='-MAXIN-')],
        [sg.Text('')],
        [sg.Text(f"Info: {img_w}x{img_h}, {frame_count} frames = {frame_count/30:.1f}sec @30fps ", p=(None,0), justification="left")],
        [sg.Text("Min value: ", p=(None,0), justification="left", key='-INFO MIN-')],
        [sg.Text("Max value: ", p=(None,0), justification="left", key='-INFO MAX-')],
        [sg.Text("Holes: ", p=(None,0), justification="left", key='-INFO HOLES-')],
        [sg.Text("Pixels clipped: ", p=(None,0), justification="left", key='-INFO CLIP-')],
        [sg.Text('Cursor: ', p=(None,0), key='-INFO CURSOR-')],
        [sg.Text('')],
        [sg.Text("Zoom    ", justification="left"),
            sg.Slider(range=(1,10), size=(30,12), default_value=1, resolution=0.1, orientation='h', key='-ZOOM-')],
        [sg.Text("Scroll X", justification="left"),
            sg.Slider(range=(0,img_w), size=(30,12), default_value=int(img_w/2), resolution=1, orientation='h', key='-SCROLLX-')],
        [sg.Text("Scroll Y", justification="left"),
            sg.Slider(range=(0,img_h), size=(30,12), default_value=int(img_h/2), resolution=1, orientation='h', key='-SCROLLY-')],
        [sg.Text("Viewport", justification="left"),
            sg.Slider(range=(0.1,2), size=(30,12), default_value=1, resolution=0.01, orientation='h', key='-SCALE-')],
        
    ]

    viewport = [
        [sg.Column([
            [sg.Text("", key="-TITLE0-")],
            [sg.Image(key="-IMAGE0-", pad=0)],
            [sg.Text("", key="-TITLE3-")],
            [sg.Image(key="-IMAGE3-", pad=0)],
            [sg.Text("", key="-TITLE6-")],
            [sg.Image(key="-IMAGE6-", pad=0)]], p=0, element_justification='center', vertical_alignment='top'),
        sg.Column([
            [sg.Text("", key="-TITLE1-")],
            [sg.Image(key="-IMAGE1-", pad=0)],
            [sg.Text("", key="-TITLE4-")],
            [sg.Image(key="-IMAGE4-", pad=0)],
            [sg.Text("", key="-TITLE7-")],
            [sg.Image(key="-IMAGE7-", pad=0)]], p=0, element_justification='center', vertical_alignment='top'),
        sg.Column([
            [sg.Text("", key="-TITLE2-")],
            [sg.Image(key="-IMAGE2-", pad=0)],
            [sg.Text("", key="-TITLE5-")],
            [sg.Image(key="-IMAGE5-", pad=0)],
            [sg.Text("", key="-TITLE8-")],
            [sg.Image(key="-IMAGE8-", pad=0)]], p=0, element_justification='center', vertical_alignment='top')]
    ]

    player_controls = [
        [sg.Slider(range=(0,frame_count-1), size=(56,8), p=0, default_value=0, resolution=1, orientation='h', disable_number_display=False, enable_events=True, key='-BAR-')],
        [sg.Button('-1', key='-PREV-'), sg.Button('Play', key='-PLAY-'), sg.Button('+1', key='-NEXT-')],
        [sg.Button('SAVE FRAME', key='-SAVE-')],
    ]

    player_col = [
        [sg.Column(viewport, p=0, justification='center')],
        [sg.Column(player_controls, p=0, element_justification='center')],
    ]

    layout = [
        [sg.Column(controls_col, p=(10,None)), sg.Column(player_col, p=0)]
    ]

    window = sg.Window("Viewer", layout, ttk_theme='clam', margins=(0,0), return_keyboard_events=False)

    _, _ = window.read(timeout=0)

    window['-MININ-'].bind('<Enter>', '+MOUSE ENTER+')
    window['-MININ-'].bind('<Leave>', '+MOUSE LEAVE+')
    window['-MININ-'].bind('<Return>', '+RETURN+')
    window['-MAXIN-'].bind('<Enter>', '+MOUSE ENTER+')
    window['-MAXIN-'].bind('<Leave>', '+MOUSE LEAVE+')
    window['-MAXIN-'].bind('<Return>', '+RETURN+')

    window.bind('<Button-1>', '+MOUSE CLICK+')

    window['-IMAGE0-'].set_focus()
    for i in range(9):
        window[f'-IMAGE{i}-'].bind('<Motion>', '+MOTION+')
        window[f'-IMAGE{i}-'].bind('<Leave>', '+LEAVE+')

    return window


def _parse_viewer_input(source, pattern, allow_dict=True):
    if isinstance(source, str) or isinstance(source, pathlib.Path):
        # can be folder or filename
        path = pathlib.Path(source)
        if path.is_dir():
            source_images = {'Untitled': sorted(list(path.glob(pattern)))}
        else:
            source_images = {'Untitled': [path]}
    elif isinstance(source, np.ndarray):
        # image array of multiple images
        source_images = {'Untitled': [source]}
    elif isinstance(source, list) or isinstance(source, tuple):
        # list of image arrays filename (or a mix)
        source_images = {'Untitled': source}
    elif allow_dict and isinstance(source, dict):
        # for multi viewport, each key relates to a viewport, to a max of 9 keys
        source_images = {}
        for key, value in source.items():
            new_data = _parse_viewer_input(value, pattern, allow_dict=False)
            source_images[key] = new_data['Untitled']
    else:
        raise TypeError('source must be a folder or file path, an image array, or list of images/filenames')
    
    return source_images

def run_gui(source, pattern='*.png'):
    '''
        PUBLIC
        View depth images with GUI
    '''
    fps = 30
    ms = 1 / fps * 1000

    source_images = _parse_viewer_input(source, pattern)

    num_ports = len(source_images)

    first_image_set = next(iter(source_images.items()))[1]
    frame_num = 0
    frame_count = len(first_image_set)
    if frame_count == 0:
        raise RuntimeError('no data/frames found to view')
    
    if isinstance(first_image_set[0], np.ndarray):
        test_image = first_image_set[0]
    else:
        test_image = cv2.imread(str(first_image_set[0]), cv2.IMREAD_UNCHANGED)
    if len(test_image.shape) != 2:
        raise ValueError('Invalid image format')

    img_h, img_w = test_image.shape
    if (np.issubdtype(test_image.dtype, np.floating)):
        type_min = 0
        type_max = 1
    else:
        type_min = np.iinfo(test_image.dtype).min
        type_max = np.iinfo(test_image.dtype).max
    
    playing = False
    in_minin = False
    in_maxin = False
    mouse_pos = None

    window = _create_viewer_window((img_w, img_h), 1, (type_min, type_max), frame_count)

    while True:
        event, values = window.read(timeout=ms)

        if event == "Exit" or event == sg.WIN_CLOSED: break

        elif event == '-MININ-+MOUSE ENTER+': in_minin = True
        elif event == '-MAXIN-+MOUSE ENTER+': in_maxin = True
        elif event == '-MININ-+MOUSE LEAVE+': in_minin = False
        elif event == '-MAXIN-+MOUSE LEAVE+': in_maxin = False

        elif event == '-IMAGE0-+LEAVE+': mouse_pos = None
        elif event == '-IMAGE1-+LEAVE+': mouse_pos = None
        elif event == '-IMAGE2-+LEAVE+': mouse_pos = None
        elif event == '-IMAGE3-+LEAVE+': mouse_pos = None
        elif event == '-IMAGE4-+LEAVE+': mouse_pos = None
        elif event == '-IMAGE5-+LEAVE+': mouse_pos = None
        elif event == '-IMAGE6-+LEAVE+': mouse_pos = None
        elif event == '-IMAGE7-+LEAVE+': mouse_pos = None
        elif event == '-IMAGE8-+LEAVE+': mouse_pos = None

        elif event == '-IMAGE0-+MOTION+':
            mouse_pos = (window['-IMAGE0-'].user_bind_event.x, window['-IMAGE0-'].user_bind_event.y)
        elif event == '-IMAGE1-+MOTION+':
            mouse_pos = (window['-IMAGE1-'].user_bind_event.x, window['-IMAGE1-'].user_bind_event.y)
        elif event == '-IMAGE2-+MOTION+':
            mouse_pos = (window['-IMAGE2-'].user_bind_event.x, window['-IMAGE2-'].user_bind_event.y)
        elif event == '-IMAGE3-+MOTION+':
            mouse_pos = (window['-IMAGE3-'].user_bind_event.x, window['-IMAGE3-'].user_bind_event.y)
        elif event == '-IMAGE4-+MOTION+':
            mouse_pos = (window['-IMAGE4-'].user_bind_event.x, window['-IMAGE4-'].user_bind_event.y)
        elif event == '-IMAGE5-+MOTION+':
            mouse_pos = (window['-IMAGE5-'].user_bind_event.x, window['-IMAGE5-'].user_bind_event.y)
        elif event == '-IMAGE6-+MOTION+':
            mouse_pos = (window['-IMAGE6-'].user_bind_event.x, window['-IMAGE6-'].user_bind_event.y)
        elif event == '-IMAGE7-+MOTION+':
            mouse_pos = (window['-IMAGE7-'].user_bind_event.x, window['-IMAGE7-'].user_bind_event.y)
        elif event == '-IMAGE8-+MOTION+':
            mouse_pos = (window['-IMAGE8-'].user_bind_event.x, window['-IMAGE8-'].user_bind_event.y)
        
        elif event == '-MININ-+RETURN+' or event == '+MOUSE CLICK+' and not in_minin:
            val = _sanitize_input_val(values["-MININ-"], type_min, int(values['-MAX-'])-1)
            values['-MIN-'] = val
            window["-MININ-"].update(value=str(val))
            window['-MIN-'].update(value=val)
        elif event == '-MAXIN-+RETURN+' or event == '+MOUSE CLICK+' and not in_maxin:
            val = _sanitize_input_val(values["-MAXIN-"], int(values['-MIN-'])+1, type_max)
            values['-MAX-'] = val
            window["-MAXIN-"].update(value=str(val))
            window['-MAX-'].update(value=val)

        elif event == '-MIN-':
            val = int(min(values['-MIN-'], values['-MAX-']-1))
            values['-MIN-'] = val
            window["-MIN-"].update(value=val)
            window["-MININ-"].update(value=str(val))
        elif event == '-MAX-':
            val = int(max(values['-MAX-'], values['-MIN-']+1))
            values['-MAX-'] = val
            window["-MAX-"].update(value=val)
            window["-MAXIN-"].update(value=str(val))

        elif event == '-SCALE-':
            mouse_pos = None

        elif event == '-BAR-':
            frame_num = int(values['-BAR-'])

        elif event == '-PLAY-':
            playing = not playing
            window["-PLAY-"].update(text='Stop' if playing else 'Play')
            window['-PREV-'].update(disabled=playing)
            window['-NEXT-'].update(disabled=playing)
            window['-SAVE-'].update(disabled=playing)
        elif event == '-PREV-': frame_num = (frame_num - 1) % frame_count
        elif event == '-NEXT-': frame_num = (frame_num + 1) % frame_count
        elif event == '-SAVE-': sred.utils.write_img(frame)

        if playing:
            frame_num = (frame_num + 1) % frame_count
        
        center = (int(values['-SCROLLX-']), int(values['-SCROLLY-']))
        val_range = (int(values['-MIN-']), int(values['-MAX-']))

        for i, key in enumerate(source_images):
            if isinstance(source_images[key][frame_num], np.ndarray):
                original = source_images[key][frame_num]
            else:
                original = cv2.imread(str(source_images[key][frame_num]), cv2.IMREAD_UNCHANGED)
            if len(original.shape) != 2:
                raise RuntimeError(f'error reading depth image, wrong number of dimensions')
            
            s = values['-SCALE-']

            fitted = sred.utils.transform(original, center, values['-ZOOM-'])
            rescaled = cv2.resize(fitted, dsize=(0,0), fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
            frame = sred.utils.colorize(rescaled, val_range, values['-CLIP-'], values['-HOLES-'])
            
            frame_BGR = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            imgbytes = cv2.imencode(".png", frame_BGR)[1].tobytes()
            if num_ports == 4 and i > 1:
                window[f"-IMAGE{i+1}-"].update(data=imgbytes)
                window[f"-TITLE{i+1}-"].update(key)
            else:
                window[f"-IMAGE{i}-"].update(data=imgbytes)
                window[f"-TITLE{i}-"].update(key)
            
            window['-BAR-'].update(value=frame_num)
            _update_frame_info(window, original, values['-ZOOM-'], center, val_range)
            
            _update_mouse_hover_info(window, rescaled, mouse_pos, val_range)

    window.close()





def _main():
    parser = argparse.ArgumentParser(description='Custom depth video viewer.')
    parser.add_argument('folder', metavar='folder', type=str,
                            help='folder containing the video frames')
    parser.add_argument('-p', '--pattern', type=str, default='*.png',
                            help='filename pattern for the files inside folder')
    
    args = parser.parse_args()
    
    folder = pathlib.Path(args.folder)
    
    run_gui(folder, args.pattern)


if __name__ == '__main__':
    _main()