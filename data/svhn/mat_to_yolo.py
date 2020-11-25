import h5py
import cv2

def get_name(index, hdf5_data):
    name_ref = hdf5_data['/digitStruct/name'][index].item()
    return ''.join([chr(v[0]) for v in hdf5_data[name_ref]])


def get_bbox(index, hdf5_data):
    attrs = {}
    item_ref = hdf5_data['/digitStruct/bbox'][index].item()
    for key in ['label', 'left', 'top', 'width', 'height']:
        attr = hdf5_data[item_ref][key]
        values = [hdf5_data[attr[i].item()][0][0].astype(int)
                  for i in range(len(attr))] if len(attr) > 1 else [attr[0][0]]
        attrs[key] = values
    return attrs

if __name__ == "__main__":
    with h5py.File('extra/digitStruct.mat') as hdf5_data:
        for i in range(3000):
            img_name = get_name(i, hdf5_data)
            print(img_name)
            im = cv2.imread('extra/'+img_name)
            h, w, c = im.shape
            arr = get_bbox(i, hdf5_data)
            

            fp = open('extra/labels/'+img_name.replace('.png','.txt'), 'w')
            arr_l = len(arr['label'])
            for idx in range(arr_l):
                label = arr['label'][idx]
                if label==10:
                    label = 0
                _l = arr['left'][idx]
                _t = arr['top'][idx]
                _w = arr['width'][idx]
                if (_l+_w)>w:
                    _w = w-_l-1
                _h = arr['height'][idx]
                if (_t+_h)>h:
                    _h = h-_t-1
                # print(w, h, _l, _t, _w , _h)
                x_center = (_l + _w/2)/w
                y_center = (_t + _h/2)/h
                bbox_width = _w/w
                bbox_height = _h/h
                # print(label, x_center, y_center, bbox_width, bbox_height)
                s = str(label)+' '+str(x_center)+' '+str(y_center)+' '+str(bbox_width)+' '+str(bbox_height)
                if idx!=(arr_l-1):
                    s += '\n'
                fp.write(s)
            fp.close()

                
