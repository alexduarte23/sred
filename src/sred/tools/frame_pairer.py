import cv2
import numpy as np
import os
import argparse
import PySimpleGUI as sg
import pathlib
import time
import tkinter as tk


def _create_pairs(filepaths, threshold=None, window=None, pairing_window=2, verbose=False):
    frame_count = len(filepaths)
    #pairs = np.zeros((frame_count-1, 4))
    pairs = np.zeros(((frame_count-pairing_window+1)*(pairing_window-1), 4))
    k = 0

    for i in range(frame_count - pairing_window + 1):
        for j in range(i+1, i+pairing_window):
            img_1 = cv2.imread(str(filepaths[i]), cv2.IMREAD_UNCHANGED)
            img_2 = cv2.imread(str(filepaths[j]), cv2.IMREAD_UNCHANGED)

            metric = rmse(img_1, img_2)
            if threshold == None: good_pair = -1
            else: good_pair = metric <= threshold

            #pairs[i] = [i, i+1, metric, good_pair]
            pairs[k] = [i, j, metric, good_pair]
            k += 1

            if k % 40 == 0:
                progress = (i+1)/(frame_count-1)*100
                if window != None:
                    window['-LOAD-'].update(value=f'Computing... {progress:.1f}%')
                    window.refresh()
                if verbose:
                    print(f'Computing... {progress:.1f}%', end='\r')
    
    if verbose:
        print('Computing... 100% ')
    
    return pairs

def _update_pairing_state(pairs, threshold):
    for pair_info in pairs:
        pair_info[3] = pair_info[2] <= threshold
    return pairs

def _update_pairing_list(window, pairs, threshold=None, order='ID'):
    if threshold != None:
        _update_pairing_state(pairs, threshold)
    
    if order == 'RMSE':
        #indexed_pairs = np.zeros((pairs.shape[0], 5))
        #indexed_pairs[:,0] = np.arange(pairs.shape[0])
        #indexed_pairs[:,1:] = pairs
        #indexed_pairs = indexed_pairs[np.argsort(indexed_pairs[:, 3])]
        #displayed_list = [f'{p[0]}: pair=<{p[1]:.0f},{p[2]:.0f}>   RMSE={p[3]:.3f}   {"GOOD" if p[4]==1 else "BAD"}' for p in indexed_pairs]

        order = np.argsort(pairs[:, 2])
        sorted_pairs = pairs[order]
        displayed_list = [f'{i}: pair=<{p[0]:.0f},{p[1]:.0f}>   RMSE={p[2]:.3f}   {"GOOD" if p[3]==1 else "BAD"}' for i, p in zip(order, sorted_pairs)]
    else:
        displayed_list = [f'{i}: pair=<{p[0]:.0f},{p[1]:.0f}>   RMSE={p[2]:.3f}   {"GOOD" if p[3]==1 else "BAD"}' for i, p in enumerate(pairs)]

    #selected = window['-PAIRING LIST-'].get_indexes()
    #window['-PAIRING LIST-'].update(values=displayed_list, set_to_index=selected)
    selected = window['-PAIRING LIST-'].get()
    if selected != []:
        pair_id = int(selected[0].split(':', 1)[0])
        selected_text = f'{pair_id}: pair=<{pairs[pair_id,0]:.0f},{pairs[pair_id,1]:.0f}>   RMSE={pairs[pair_id,2]:.3f}   {"GOOD" if pairs[pair_id,3]==1 else "BAD"}'
        window['-PAIRING LIST-'].update(values=displayed_list)
        window['-PAIRING LIST-'].set_value([selected_text])
    else:
        window['-PAIRING LIST-'].update(values=displayed_list)
    
    good = np.count_nonzero(pairs[:,3]==1)
    if threshold != None:
        window['-GOOD-'].update(value=f'Good pairs: {good} of {pairs.shape[0]} ({good/pairs.shape[0]*100:.2f}%)')

    return displayed_list, pairs, good

def _inside_element_box(window, name):
    pointer_x = window.TKroot.winfo_pointerx()
    pointer_y = window.TKroot.winfo_pointery()
    x = window[name].Widget.winfo_rootx()
    y = window[name].Widget.winfo_rooty()
    w, h = window[name].get_size()

    return x <= pointer_x <= x+w and y <= pointer_y <= y+h

def _save_to_file(filename, pairs, filepaths):
    if filename == '' or filename == None: return
    
    with open(filename, "w") as f:
        data = [f'{filepaths[int(p[0])]} : {filepaths[int(p[1])]}\n' for p in pairs if p[3]==1]
        data[-1] = data[-1][:-1]
        f.writelines(data)


def _sanitize_thresh_val(input_str, min_val, max_val):
    if input_str == '' or input_str == None:
        return max(min(0, max_val), min_val)

    sign = -1 if input_str[0] == '-' else 1

    split = input_str.split('.', 1)
    if len(split) == 2:
        whole, decimal = split
    else:
        whole, decimal = split[0], ''
        
    whole_digits = ''.join(c for c in whole if c.isdigit())
    decimal_digits = ''.join(c for c in decimal if c.isdigit())
    str_num = whole_digits + '.' + decimal_digits
    
    if str_num == '.': str_num = '0'

    num = sign * float(str_num)
    return  max(min(num, max_val), min_val)


def _create_app_window(thresh_range, pairing_window):

    sg.theme('SystemDefault')

    top_section = [
        [sg.Text("Threshold")],
        [sg.Slider(range=(thresh_range[0], thresh_range[1]), size=(90,12), default_value=thresh_range[0], resolution=0.001, orientation='h', disable_number_display=True, enable_events=True, key='-THRESH-'), 
            sg.Input(str(thresh_range[0]), size=(12,None), key='-THRESH IN-')],
    ]

    list_section = [
        [sg.Text("Good pairs: ?", key='-GOOD-')],
        [sg.Text("")],
        [sg.Text("Sort by:"),
            sg.Radio('ID', group_id=1, default=True, enable_events=True, key='-SORT ID-'),
            sg.Radio('RMSE', group_id=1, default=False, enable_events=True, key='-SORT RMSE-')],
        [sg.Listbox(values=['Computing...'], disabled=True, enable_events=True, size=(50, 20), key="-PAIRING LIST-")],
        [sg.Button('SAVE GOOD PAIRS', key='-SAVE-')]
    ]

    preview_section = [
        [sg.Text("Preview Pairs")],
        [sg.Radio('1st', group_id=0, default=False, enable_events=True, key='-FIRST-'),
            sg.Radio('2nd', group_id=0, default=False, enable_events=True, key='-SECOND-'),
            sg.Radio('Flicker', group_id=0, default=True, enable_events=True, key='-FLICKER-')],
        [sg.Image(key="-IMAGE-")],
    ]

    max_pairing_window = max(10, pairing_window)
    layout_1 = [
        [sg.Text("Pairing Window Size")],
        [sg.Slider(range=(2, max_pairing_window), size=(30,12), default_value=pairing_window, resolution=1, orientation='h', key='-PAIRING WINDOW-')], 
        [sg.Button('COMPUTE', key='-COMPUTE-')]
    ]

    layout_2 = [[sg.Text("Computing...", p=(50,50), justification='center', key='-LOAD-')]]

    layout_3 = [
        [sg.Column(top_section)],
        [sg.Column(list_section, element_justification='center'), sg.VSeparator(), sg.Column(preview_section)]
    ]

    layout = [
        [sg.Column(layout_1, key='-START LAYOUT-'),
            sg.Column(layout_2, visible=False, key='-LOAD LAYOUT-'),
            sg.Column(layout_3, visible=False, key='-MAIN LAYOUT-')]
    ]

    window = sg.Window("Frame Pairing", layout, ttk_theme='clam')

    _, _ = window.read(timeout=0)

    window['-THRESH IN-'].bind('<Return>', '+RETURN+')
    window.bind('<Button-1>', '+MOUSE CLICK+')
    window['-IMAGE-'].set_focus()

    return window


def run_gui(folder, pattern='*.png', threshold=None, pairing_window=2, output=None):
    folder = pathlib.Path(folder)
    filepaths = sorted(list(folder.glob(pattern)))

    window = _create_app_window((0,10), pairing_window)

    event = ''
    while event != '-COMPUTE-':
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            window.close()
            return

    pairing_window = int(values['-PAIRING WINDOW-'])
    window['-START LAYOUT-'].update(visible=False)
    window['-LOAD LAYOUT-'].update(visible=True)
    
    #pairs = [compute_pairing(filepaths, i, i+1) for i in range(len(filepaths)-1)]
    #all_rmse = np.array([pair['rmse'] for pair in pairs])
    #thresh_range = (np.min(all_rmse), np.max(all_rmse))
    
    pairs = _create_pairs(filepaths, window=window, pairing_window=pairing_window)
    thresh_range = (np.min(pairs[:,2]), np.max(pairs[:,2]))
    if threshold == None:
        initial_thresh = (thresh_range[0] + thresh_range[1]) * 0.5
    else:
        initial_thresh = min(max(threshold, thresh_range[0]), thresh_range[1])

    window['-LOAD LAYOUT-'].update(visible=False)
    window['-MAIN LAYOUT-'].update(visible=True)

    window['-THRESH-'].update(range=thresh_range, value=initial_thresh)
    window['-THRESH IN-'].update(value=str(initial_thresh))

    displayed_list, pairs, good = _update_pairing_list(window, pairs, threshold=initial_thresh, order='ID')
    
    in_threshbox = False
    commit_input_thresh = False
    last_time = time.time()
    flicker_interval = 0.4
    flicker_timer = 0
    current_img_idx = 0
    update_img = False
    current_order = 'ID'

    window['-PAIRING LIST-'].update(set_to_index=0, disabled=False)
    pair_id = int(displayed_list[0].split(':', 1)[0])

    img_id = int(pairs[pair_id][current_img_idx])
    window['-IMAGE-'].update(filename=str(filepaths[img_id]))

    while True:
        event, values = window.read(timeout=100)


        if event == "Exit" or event == sg.WIN_CLOSED: break

        elif event == '+MOUSE CLICK+' and _inside_element_box(window, '-THRESH IN-'):
            in_threshbox = True
        elif event == '+MOUSE CLICK+' and in_threshbox:
            in_threshbox = False
            window['-IMAGE-'].set_focus()
            commit_input_thresh = True
        elif event == '-THRESH IN-+RETURN+':
            commit_input_thresh = True
        elif event == '-THRESH-':
            window["-THRESH IN-"].update(value=str(values['-THRESH-']))
            displayed_list, pairs, good = _update_pairing_list(window, pairs, threshold=values['-THRESH-'], order=current_order)
        
        elif event == '+MOUSE CLICK+' and _inside_element_box(window, '-PAIRING LIST-'):
            # for some reason the 'window bind click' disables the PAIRING LIST event
            selected = window['-PAIRING LIST-'].get()[0]
            pair_id = int(selected.split(':', 1)[0])
            update_img = True

        elif event == '-SORT ID-' and current_order != 'ID':
            current_order = 'ID'
            displayed_list, pairs, good = _update_pairing_list(window, pairs, order='ID')
        elif event == '-SORT RMSE-' and current_order != 'RMSE':
            current_order = 'RMSE'
            displayed_list, pairs, good = _update_pairing_list(window, pairs, order='RMSE')
        
        elif event == '-FIRST-':
            current_img_idx = 0
            update_img = True
        elif event == '-SECOND-':
            current_img_idx = 1
            update_img = True
        elif event == '-FLICKER-':
            current_img_idx = current_img_idx % 2
            update_img = True
            
        elif event == '-SAVE-':
            types = [('Text Document', '*.txt')]
            output_path = pathlib.Path('pairs') if output == None else pathlib.Path(output)
            output_dir = '.' if output_path.parent == '' else output_path.parent
            output_name = output_path.name
            filename = tk.filedialog.asksaveasfilename(filetypes=types, defaultextension=types, initialdir=output_dir, initialfile=output_name)
            _save_to_file(filename, pairs, filepaths)

        if values['-FLICKER-']:
            curr_time = time.time()
            dt = curr_time - last_time
            last_time = curr_time
            flicker_timer += dt
            if flicker_timer >= flicker_interval:
                flicker_timer = flicker_timer % flicker_interval
                current_img_idx = (current_img_idx + 1) % 2
                update_img = True

        if update_img:
            update_img = False
            img_id = int(pairs[pair_id][current_img_idx])
            window['-IMAGE-'].update(filename=str(filepaths[img_id]))

        if commit_input_thresh:
            commit_input_thresh = False
            val = _sanitize_thresh_val(values["-THRESH IN-"], thresh_range[0], thresh_range[1])
            values['-THRESH-'] = val
            window["-THRESH IN-"].update(value=str(val))
            window['-THRESH-'].update(value=val)
            displayed_list, pairs, good = _update_pairing_list(window, pairs, threshold=val, order=current_order)

    window.close()

    return good


#def rmse(im1, im2):
#    return np.sqrt(np.sum((im1 - im2)**2) / (im1.shape[0] * im1.shape[1]))

def rmse(im1, im2):
    not_holes = (im1 != 0) & (im2 != 0)
    valid_1 = im1[not_holes].astype('float32')
    valid_2 = im2[not_holes].astype('float32')

    return np.sqrt(np.sum((valid_1 - valid_2)**2) / len(valid_1))


def run_commandline(folder, pattern='*.png', threshold=None, pairing_window=2, output=None):
    folder = pathlib.Path(folder)
    filepaths = sorted(list(folder.glob(pattern)))
    
    if threshold == None:
        pairs = _create_pairs(filepaths, pairing_window=pairing_window, verbose=True)

        deduced_threshold = (1-0.4)*np.mean(pairs[:,2]) + 0.4*np.max(pairs[:,2])
        print('Deduced threshold:', deduced_threshold)
        _update_pairing_state(pairs, deduced_threshold)
    else:
        pairs = _create_pairs(filepaths, threshold, pairing_window=pairing_window, verbose=True)
    
    good = np.count_nonzero(pairs[:,3]==1)
    print(f'Good pairs: {good} of {pairs.shape[0]} ({good/pairs.shape[0]*100:.2f}%)')

    if output == None:
        output = 'goodpairs_' + str(time.time()) + '.txt'
    filename = pathlib.Path(output)
    _save_to_file(filename, pairs, filepaths)
    print('Saved to:', filename)
    
    return good


def _pairing_window_type(x):
    x = int(x)
    if x < 2:
        raise argparse.ArgumentTypeError("Minimum pairing window size is 2")
    return x

def _main():
    parser = argparse.ArgumentParser(description='Custom depth video viewer.')
    parser.add_argument('folder', type=str, help='folder containing the video frames')
    parser.add_argument('-p', '--pattern', type=str, default='*.png', help='filename pattern for the files inside folder')
    parser.add_argument('-t', '--threshold', type=float, help='threshold for determining good pairs')
    parser.add_argument('-w', '--window', type=_pairing_window_type, default=2, help='size of the pairing window (minimum=default=2 => only adjacent frames)')
    parser.add_argument('-o', '--output', type=str, help='output name to store the pairing (txt file)')
    parser.add_argument('--nogui', action='store_true', help='run GUI version')
    
    args = parser.parse_args()
    
    folder = pathlib.Path(args.folder)
    

    if args.nogui:
        run_commandline(folder, args.pattern, args.threshold, args.window, args.output)
    else:
        run_gui(folder, args.pattern, args.threshold, args.window, args.output)


if __name__ == '__main__':
    _main()