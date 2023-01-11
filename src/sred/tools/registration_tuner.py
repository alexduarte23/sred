import time
import cv2
import numpy as np
import pathlib
import PySimpleGUI as sg
import argparse
import sys

if sys.path[-1] != str(pathlib.Path(__file__).parents[1]):
    sys.path.append(str(pathlib.Path(__file__).parents[1]))

import sred.utils



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




def _create_viewer_window(img_size, type_limits, frame_count):
    ''' Create GUI window and layout '''

    img_w, img_h = img_size
    type_min, type_max = type_limits

    sg.theme('SystemDefault')

    controls_col = [
        #[sg.Slider(range=(type_min,type_max), size=(30,12), default_value=type_max, resolution=1, orientation='h', disable_number_display=True, enable_events=True, key='-MAX-'), 
        #    sg.Input(str(type_max), size=(7,None), key='-MAXIN-')],
        [sg.Column([[sg.Text('Cam Parameters', font='Any 14')]], p=0, justification='center')],
        [sg.Text("d_fx", size=5),
            sg.Slider(range=(0,600), size=(30,12), default_value=363.58, resolution=0.01, orientation='h', disable_number_display=True, enable_events=True, key='-d_fx-'), 
            sg.Input(str(363.58), size=(7,None), key='-d_fx_INP-')],
        [sg.Text("d_fy", size=5),
            sg.Slider(range=(0,600), size=(30,12), default_value=363.53, resolution=0.01, orientation='h', disable_number_display=True, enable_events=True, key='-d_fy-'), 
            sg.Input(str(363.53), size=(7,None), key='-d_fy_INP-')],
        [sg.Text("d_cx", size=5),
            sg.Slider(range=(0,500), size=(30,12), default_value=250.32, resolution=0.01, orientation='h', disable_number_display=True, enable_events=True, key='-d_cx-'), 
            sg.Input(str(250.32), size=(7,None), key='-d_cx_INP-')],
        [sg.Text("d_cy", size=5),
            sg.Slider(range=(0,500), size=(30,12), default_value=212.55, resolution=0.01, orientation='h', disable_number_display=True, enable_events=True, key='-d_cy-'), 
            sg.Input(str(212.55), size=(7,None), key='-d_cy_INP-')],
        [sg.Text('')],
        [sg.Text("rgb_fx", size=5),
            sg.Slider(range=(0,2000), size=(30,12), default_value=1054.35, resolution=0.01, orientation='h', disable_number_display=True, enable_events=True, key='-rgb_fx-'), 
            sg.Input(str(1054.35), size=(7,None), key='-rgb_fx_INP-')],
        [sg.Text("rgb_fy", size=5),
            sg.Slider(range=(0,2000), size=(30,12), default_value=1054.51, resolution=0.01, orientation='h', disable_number_display=True, enable_events=True, key='-rgb_fy-'), 
            sg.Input(str(1054.51), size=(7,None), key='-rgb_fy_INP-')],
        [sg.Text("rgb_cx", size=5),
            sg.Slider(range=(0,2000), size=(30,12), default_value=956.12, resolution=0.01, orientation='h', disable_number_display=True, enable_events=True, key='-rgb_cx-'), 
            sg.Input(str(956.12), size=(7,None), key='-rgb_cx_INP-')],
        [sg.Text("rgb_cy", size=5),
            sg.Slider(range=(0,2000), size=(30,12), default_value=548.99, resolution=0.01, orientation='h', disable_number_display=True, enable_events=True, key='-rgb_cy-'), 
            sg.Input(str(548.99), size=(7,None), key='-rgb_cy_INP-')],
        [sg.Text('')],
        [sg.Text("angle", size=5),
            sg.Slider(range=(-0.5,0.5), size=(30,12), default_value=0.00, resolution=0.01, orientation='h', disable_number_display=True, enable_events=True, key='-angle-'), 
            sg.Input(str(0.00), size=(7,None), key='-angle_INP-')],
        [sg.Text("t_x", size=5),
            sg.Slider(range=(-500,500), size=(30,12), default_value=290.00, resolution=0.01, orientation='h', disable_number_display=True, enable_events=True, key='-t_x-'), 
            sg.Input(str(290.00), size=(7,None), key='-t_x_INP-')],
        [sg.Text("t_y", size=5),
            sg.Slider(range=(-200,200), size=(30,12), default_value=0.00, resolution=0.01, orientation='h', disable_number_display=True, enable_events=True, key='-t_y-'), 
            sg.Input(str(0.00), size=(7,None), key='-t_y_INP-')],
        [sg.Text('')],

        [sg.Column([[sg.Text('View Settings', font='Any 14')]], p=0, justification='center')],
        [sg.Text("Zoom    ", justification="left"),
            sg.Slider(range=(1,10), size=(30,12), default_value=1, resolution=0.1, orientation='h', enable_events=True, key='-ZOOM-')],
        [sg.Text("Scroll X", justification="left"),
            sg.Slider(range=(0,img_w), size=(30,12), default_value=int(img_w/2), resolution=1, orientation='h', enable_events=True, key='-SCROLLX-')],
        [sg.Text("Scroll Y", justification="left"),
            sg.Slider(range=(0,img_h), size=(30,12), default_value=int(img_h/2), resolution=1, orientation='h', enable_events=True, key='-SCROLLY-')],
        [sg.Text("Viewport", justification="left"),
            sg.Slider(range=(0.1,2), size=(30,12), default_value=1, resolution=0.01, orientation='h', enable_events=True, key='-SCALE-')],
        
    ]

    player_col = [
        [sg.Text("Display:"),
            sg.Radio('Registered', group_id=0, default=True, enable_events=True, key='-DISP REG-'),
            sg.Radio('Contours', group_id=0, default=False, enable_events=True, key='-DISP CONT-'),
            sg.Radio('Overlay', group_id=0, default=False, enable_events=True, key='-DISP OVER-')],
        [sg.Image(key="-IMAGE-")],#, pad=0)],
        [sg.Slider(range=(0,frame_count-1), size=(56,8), p=0, default_value=0, resolution=1, orientation='h', disable_number_display=False, enable_events=True, key='-BAR-')],
        [sg.Button('-1', key='-PREV-'), sg.Button('Play', key='-PLAY-'), sg.Button('+1', key='-NEXT-')],
        [sg.Button('SAVE FRAME', key='-SAVE-')],
    ]

    layout = [
        [sg.Column(controls_col, p=(10,10)), sg.Column(player_col, p=(10,10), element_justification='center')]
    ]

    window = sg.Window("Registration Tuner", layout, ttk_theme='clam', margins=(0,0), return_keyboard_events=False)

    _, _ = window.read(timeout=0)

    #window['-MININ-'].bind('<Enter>', '+MOUSE ENTER+')
    #window['-MININ-'].bind('<Leave>', '+MOUSE LEAVE+')
    #window['-MININ-'].bind('<Return>', '+RETURN+')

    window.bind('<Button-1>', '+MOUSE CLICK+')

    window['-IMAGE-'].set_focus()

    return window


def _parse_viewer_input(source, pattern):
    if isinstance(source, str) or isinstance(source, pathlib.Path):
        # can be folder or filename
        path = pathlib.Path(source)
        if path.is_dir():
            source_images = sorted(list(path.glob(pattern)))
        else:
            source_images = [path]
    elif isinstance(source, np.ndarray):
        # image array of multiple images
        source_images = [source]
    elif isinstance(source, list) or isinstance(source, tuple):
        # list of image arrays filename (or a mix)
        source_images = source
    else:
        raise TypeError('source must be a folder or file path, an image array, or list of images/filenames')
    
    return source_images

def run_gui(d_source, rgb_source, pattern='*.png'):
    '''
        PUBLIC
        View depth images with GUI
    '''
    fps = 30
    ms = 1 / fps * 1000

    d_source_images = _parse_viewer_input(d_source, pattern)
    rgb_source_images = _parse_viewer_input(rgb_source, pattern)

    if len(d_source_images) != len(rgb_source_images):
        raise RuntimeError('Number of depth and RGB frames not matching')

    frame_count = len(d_source_images)
    if frame_count == 0:
        raise RuntimeError('no data/frames found to view')
    
    # validate and sample first image before creating gui
    test_image = d_source_images[0]
    if not isinstance(test_image, np.ndarray):
        test_image = cv2.imread(str(test_image), cv2.IMREAD_UNCHANGED)
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
    frame_num = 0
    update_image = True # update all
    update_port = True # keep reg

    cam_params_orig = {
        # intrinsics
        'd_fx': 363.58,
        'd_fy': 363.53,
        'd_cx': 250.32,
        'd_cy': 212.55,
        'rgb_fx': 1054.35,
        'rgb_fy': 1054.51,
        'rgb_cx': 956.12,
        'rgb_cy': 548.99,
        # extrinsics
        'angle': 0,
        't_x': 290,
        't_y': -10 +10
    }

    cam_params = cam_params_orig.copy()

    window = _create_viewer_window((img_w, img_h), (type_min, type_max), frame_count)

    while True:
        event, values = window.read(timeout=ms)

        if event == "Exit" or event == sg.WIN_CLOSED: break

        #elif event == '-MININ-+MOUSE ENTER+': in_minin = True
        #elif event == '-MININ-+MOUSE LEAVE+': in_minin = False
        #elif event == '-MININ-+RETURN+' or event == '+MOUSE CLICK+' and not in_minin:
        #    val = _sanitize_input_val(values["-MININ-"], type_min, int(values['-MAX-'])-1)
        #    values['-MIN-'] = val
        #    window["-MININ-"].update(value=str(val))
        #    window['-MIN-'].update(value=val)
        #elif event == '-MIN-':
        #    val = int(min(values['-MIN-'], values['-MAX-']-1))
        #    values['-MIN-'] = val
        #    window["-MIN-"].update(value=val)
        #    window["-MININ-"].update(value=str(val))
        
        #elif event in ('-d_fx_INP-', '-d_fy_INP-', '-d_cx_INP-', '-d_cy_INP-', '-rgb_fx_INP-', '-rgb_fy_INP-', '-rgb_cx_INP-', '-rgb_cy_INP-', '-angle_INP-', '-t_x_INP-', '-t_y_INP-'):
        #    val = _sanitize_input_val(values["-MININ-"], type_min, int(values['-MAX-'])-1)
        #    values['-MIN-'] = val
        #    window["-MININ-"].update(value=str(val))
        #    window['-MIN-'].update(value=val)
        elif event in ('-d_fx-', '-d_fy-', '-d_cx-', '-d_cy-', '-rgb_fx-', '-rgb_fy-', '-rgb_cx-', '-rgb_cy-', '-angle-', '-t_x-', '-t_y-'):
            cam_params[event[1:-1]] = values[event]
            window[event[:-1]+'_INP-'].update(value=str(values[event]))
            update_image = True

        elif event == '-BAR-':
            frame_num = int(values['-BAR-'])
            update_image = True

        elif event == '-PLAY-':
            playing = not playing
            window["-PLAY-"].update(text='Stop' if playing else 'Play')
            window['-PREV-'].update(disabled=playing)
            window['-NEXT-'].update(disabled=playing)
            window['-SAVE-'].update(disabled=playing)
        elif event == '-PREV-':
            frame_num = (frame_num - 1) % frame_count
            update_image = True
        elif event == '-NEXT-':
            frame_num = (frame_num + 1) % frame_count
            update_image = True
        elif event == '-SAVE-':
            sred.utils.write_img(reg_rgb)
            update_image = True
        elif event in ('-SCALE-', '-SCROLLX-', '-SCROLLY-', '-ZOOM-'):
            update_port = True
        elif event in ('-DISP REG-', '-DISP CONT-', '-DISP OVER-'):
            update_port = True

        if playing:
            frame_num = (frame_num + 1) % frame_count
            update_image = True
        
        if update_image or update_port:
            if update_image:
                if isinstance(d_source_images[frame_num], np.ndarray):
                    d_img = d_source_images[frame_num]
                    rgb_img = rgb_source_images[frame_num]
                else:
                    d_img = cv2.imread(str(d_source_images[frame_num]), cv2.IMREAD_UNCHANGED)
                    rgb_img = cv2.imread(str(rgb_source_images[frame_num]), cv2.IMREAD_UNCHANGED)
                if len(d_img.shape) != 2 or len(rgb_img.shape) != 3:
                    raise RuntimeError(f'error reading image, wrong number of dimensions')

                filled = sred.utils.hole_interpolation(d_img)
                blured = sred.utils.smoothen_filled_holes(d_img, filled)
                reg_rgb = sred.utils.register_rgb(rgb_img, blured, cam_params)
            
            
            if values['-DISP REG-']:
                disp_img = reg_rgb
            elif values['-DISP CONT-']:
                imgray = cv2.cvtColor(reg_rgb, cv2.COLOR_RGB2GRAY)
                canny_output = cv2.Canny(imgray, 170, 200)
                contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                norm_d_img = cv2.cvtColor(d_img, cv2.COLOR_GRAY2RGB).astype('float64') / np.iinfo('uint16').max
                disp_img = (norm_d_img * 255).astype('uint8')
                cv2.drawContours(disp_img, contours, -1, (0,255,0), 1)
            elif values['-DISP OVER-']:
                norm_d_img = cv2.cvtColor(d_img, cv2.COLOR_GRAY2RGB).astype('float64') / np.iinfo('uint16').max
                norm_reg_rgb = reg_rgb.astype('float64') / np.iinfo('uint8').max
                disp_img = ((0.5*norm_d_img + 0.5*norm_reg_rgb) * 255).astype('uint8')

            s = values['-SCALE-']
            center = (int(values['-SCROLLX-']), int(values['-SCROLLY-']))

            fitted = sred.utils.transform(disp_img, center, values['-ZOOM-'])
            rescaled = cv2.resize(fitted, dsize=(0,0), fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
            imgbytes = cv2.imencode(".png", rescaled)[1].tobytes()
            window[f"-IMAGE-"].update(data=imgbytes)
            
            window['-BAR-'].update(value=frame_num)
            update_image = False
            update_port = False


    window.close()




def _main():
    parser = argparse.ArgumentParser(description='Custom depth video viewer.')
    parser.add_argument('d_folder', metavar='d_folder', type=str,
                            help='folder containing the depth video frames')
    parser.add_argument('rgb_folder', metavar='rgb_folder', type=str,
                            help='folder containing the RGB video frames')
    parser.add_argument('-p', '--pattern', type=str, default='*.png',
                            help='filename pattern for the files inside folder')
    
    args = parser.parse_args()
    
    d_folder = pathlib.Path(args.d_folder)
    rgb_folder = pathlib.Path(args.rgb_folder)
    
    run_gui(d_folder, rgb_folder, args.pattern)


if __name__ == '__main__':
    _main()