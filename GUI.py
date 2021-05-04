from tkinter import *
from tkinter import colorchooser, simpledialog, ttk
from PIL import Image, ImageTk, ImageDraw
import tkinter.filedialog as tkFileDialog
import numpy as np
import argparse
import cv2
import tensorflow as tf
import subprocess
import os
from config_options import ConfigOptions
from ttkthemes import ThemedStyle
import tkinter as tk
from network import GMCNNModel

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tf.reset_default_graph()

class Paint(object):
    def __init__(master, config):
        master.config = config
        master.root = tk.Tk()
        master.root.title('Image Inpainter')
        master.root['background']='#cfcfcf'
        
        style = ThemedStyle(master.root)
        style.set_theme("plastik")

        master.pen_button = ttk.Button(master.root, text='Pen', command=master.use_pen)
        master.rect_button = ttk.Button(master.root, text='Rectangle', command=master.use_rect)
        master.pen_size_button = ttk.Button(master.root, text='Pen Size', command=master.pen_size)
        master.color_picker_button = ttk.Button(master.root, text='Drawing\n  Color', command=master.color_picker)
        master.c = Canvas(master.root, bg='white', width=config.img_shapes[1], height=config.img_shapes[0])
        master.out = Canvas(master.root, bg='white', width=config.img_shapes[1], height=config.img_shapes[0])
        master.inpaint_button = ttk.Button(master.root, text='Inpaint', command=master.inpaint)
        master.clear_button = ttk.Button(master.root, text='Clear Shape', command=master.clear)
        master.open_button = ttk.Button(master.root, text='Open', command=master.open)
        master.save_button = ttk.Button(master.root, text="Save", command=master.save)

        master.pen_button.grid(row=0, column=0, padx=5, ipady=20, ipadx=10)
        master.rect_button.grid(row=1, column=0, padx=5, ipady=20, ipadx=10)
        master.pen_size_button.grid(row=6, column=0, padx=5, ipady=20, ipadx=10)
        master.color_picker_button.grid(row=7, column=0, padx=5, ipady=20, ipadx=10)
        master.c.grid(row=0, column=1, rowspan=8)
        master.out.grid(row=0, column=2, rowspan=8)
        master.inpaint_button.grid(row=0, column=3, padx=5, ipady=20, ipadx=10)
        master.clear_button.grid(row=1, column=3, padx=5, ipady=20, ipadx=10)
        master.open_button.grid(row=6, column=3, padx=5, ipady=20, ipadx=10)
        master.save_button.grid(row=7, column=3, padx=5, ipady=20, ipadx=10)

        master.filename = None
        master.setup()
        master.root.mainloop()

    def setup(master):
        master.old_x = None
        master.old_y = None
        master.start_x = None
        master.start_y = None
        master.end_x = None
        master.end_y = None
        master.rect_buf = None
        master.pen_buf = None
        master.im_h = None
        master.im_w = None
        master.mask = None
        master.result = None
        master.blank = None
        master.eraser_on = False
        master.isPainting = False
        master.active_button = master.rect_button
        master.c.bind('<B1-Motion>', master.paint)
        master.c.bind('<ButtonRelease-1>', master.reset)
        master.c.bind('<Button-1>', master.beginPaint)
        master.c.bind('<Enter>', master.icon2pen)
        master.c.bind('<Leave>', master.icon2mice)
        master.mode = 'rect'
        assert master.mode in ['rect', 'pen']
        master.paint_color = 'red'
        master.mask_candidate = []
        master.rect_candidate = []
        master.pen_width = 20

        master.model = GMCNNModel()
        master.reuse = False
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = False
        master.sess = tf.Session(config=sess_config)

        master.input_image_tf = tf.placeholder(dtype=tf.float32,
                                             shape=[1, master.config.img_shapes[0], master.config.img_shapes[1], 3])
        master.input_mask_tf = tf.placeholder(dtype=tf.float32,
                                            shape=[1, master.config.img_shapes[0], master.config.img_shapes[1], 1])

        output = master.model.evaluate(master.input_image_tf, master.input_mask_tf, config=master.config, reuse=master.reuse)
        output = (output + 1) * 127.5
        output = tf.minimum(tf.maximum(output[:, :, :, ::-1], 0), 255)
        master.output = tf.cast(output, tf.uint8)

        # load pretrained model
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = list(map(lambda x: tf.assign(x, tf.contrib.framework.load_variable(config.load_model_dir, x.name)),
                              vars_list))
        master.sess.run(assign_ops)

    def checkResp(master):
        assert len(master.mask_candidate) == len(master.rect_candidate)

    def open(master):
        master.filename = tkFileDialog.askopenfilename(initialdir='./imgs',
                                                     title="Select a file",
                                                     filetypes=(("png and jpg files", "*.png *.jpg *.jpeg"),
                                                                ("all files", "*.*")))
        master.filename_ = master.filename.split('/')[-1][:-4]
        master.filepath = '/'.join(master.filename.split('/')[:-1])
        print(master.filename_, master.filepath)
        master.activate_button(master.pen_button)
        master.mode = 'pen'
        try:
            photo = Image.open(master.filename)
            master.image = cv2.imread(master.filename)
        except:
            print('exception')
        else:
            _extracted_from_open_17(master, photo)

    def open_sub_function(master, photo):
        master.im_w, master.im_h = photo.size
        master.mask = np.zeros((master.im_h, master.im_w, 1)).astype(np.uint8)
        print(photo.size)
        master.displayPhoto = photo
        master.displayPhoto = master.displayPhoto.resize((master.im_w, master.im_h))
        master.draw = ImageDraw.Draw(master.displayPhoto)
        master.photo_tk = ImageTk.PhotoImage(image=master.displayPhoto)
        master.c.create_image(0, 0, image=master.photo_tk, anchor=NW)
        master.rect_candidate.clear()
        master.mask_candidate.clear()
        if master.blank is None:
            master.blank = Image.open('blank.png')
        master.blank = master.blank.resize((master.im_w, master.im_h))
        master.blank_tk = ImageTk.PhotoImage(image=master.blank)
        master.out.create_image(0, 0, image=master.blank_tk, anchor=NW)

    def save(master):
        file = tkFileDialog.asksaveasfile(mode='w', title='Save Image', defaultextension=".png", filetypes = (("png files","*.png"),("all files","*.*")))
        cv2.imwrite(file.name, master.result[0][:, :, ::-1])

    def inpaint(master):
        if master.mode == 'rect':
            for rect in master.mask_candidate:
                master.mask[rect[1]:rect[3], rect[0]:rect[2], :] = 1
        image = np.expand_dims(master.image, 0)
        mask = np.expand_dims(master.mask, 0)
        print(image.shape)
        print(mask.shape)   

        master.result = master.sess.run(master.output, feed_dict={master.input_image_tf: image * 1.0,
                                                            master.input_mask_tf: mask * 1.0})
        cv2.imwrite('./tmp.png', master.result[0][:, :, ::-1])

        photo = Image.open('./tmp.png')
        
        master.displayPhotoResult = photo
        master.displayPhotoResult = master.displayPhotoResult.resize((master.im_w, master.im_h))
        master.photo_tk_result = ImageTk.PhotoImage(image=master.displayPhotoResult)
        master.out.create_image(0, 0, image=master.photo_tk_result, anchor=NW)
        return

    def use_rect(master):
        master.activate_button(master.rect_button)
        master.mode = 'rect'

    def use_pen(master):
        master.activate_button(master.pen_button)
        master.mode = 'pen'

    def clear(master):
        master.mask = np.zeros((master.im_h, master.im_w, 1)).astype(np.uint8)
        if master.mode == 'pen':
            _extracted_from_clear_4(master)
        else:
            if master.rect_candidate is None or len(master.rect_candidate) == 0:
                return
            for item in master.rect_candidate:
                master.c.delete(item)
            master.rect_candidate.clear()
            master.mask_candidate.clear()
            master.checkResp()

    def clea_sub_function(master):
        photo = Image.open(master.filename)
        master.image = cv2.imread(master.filename)
        master.displayPhoto = photo
        master.displayPhoto = master.displayPhoto.resize((master.im_w, master.im_h))
        master.draw = ImageDraw.Draw(master.displayPhoto)
        master.photo_tk = ImageTk.PhotoImage(image=master.displayPhoto)
        master.c.create_image(0, 0, image=master.photo_tk, anchor=NW)
            
    def color_picker(master):
        master.activate_button(master.color_picker_button)
        colorPicked = colorchooser.askcolor(title ="Choose color")[1]
        master.paint_color = colorPicked

    def pen_size(master):
        master.activate_button(master.pen_button)
        master.pen_width = simpledialog.askinteger("Pen Size", "Current Size %s" % (master.pen_width))

    def activate_button(master, some_button, eraser_mode=False):
        master.active_button.state(['!pressed'])
        some_button.state(['pressed'])
        master.active_button = some_button
        master.eraser_on = eraser_mode

    def beginPaint(master, event):
        master.start_x = event.x
        master.start_y = event.y
        master.isPainting = True

    def paint(master, event):
        if master.start_x and master.start_y and master.mode == 'rect':
            master.end_x = max(min(event.x, master.im_w), 0)
            master.end_y = max(min(event.y, master.im_h), 0)
            rect = master.c.create_rectangle(master.start_x, master.start_y, master.end_x, master.end_y, fill=master.paint_color)
            if master.rect_buf is not None:
                master.c.delete(master.rect_buf)
            master.rect_buf = rect
        elif master.old_x and master.old_y and master.mode == 'pen':
            pen_draw = master.c.create_line(master.old_x, master.old_y, event.x, event.y,
                                      width=master.pen_width, fill=master.paint_color, capstyle=ROUND,
                                      smooth=True, splinesteps=36)
            cv2.line(master.mask, (master.old_x, master.old_y), (event.x, event.y), (1), master.pen_width)
        master.old_x = event.x
        master.old_y = event.y

    def reset(master, event):
        master.old_x, master.old_y = None, None
        if master.mode == 'rect':
            master.isPainting = False
            rect = master.c.create_rectangle(master.start_x, master.start_y, master.end_x, master.end_y,
                                           fill=master.paint_color)
            if master.rect_buf is not None:
                master.c.delete(master.rect_buf)
            master.rect_buf = None
            master.rect_candidate.append(rect)

            x1, y1, x2, y2 = min(master.start_x, master.end_x), min(master.start_y, master.end_y),\
                             max(master.start_x, master.end_x), max(master.start_y, master.end_y)
            master.mask_candidate.append((x1, y1, x2, y2))
            print(master.mask_candidate[-1])

    def icon2pen(master, event):
        return

    def icon2mice(master, event):
        return


if __name__ == '__main__':
    config = ConfigOptions().parse()
    config.mode = 'silent'
    ge = Paint(config)
