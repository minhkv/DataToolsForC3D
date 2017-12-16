from create_of_image import *



def test_of_image(u_path, v_path, flow_path, number_result=False):
    flow_u = read_image(u_path, 0)
    flow_v = read_image(v_path, 0)
    flow = read_image(flow_path, 1)
    
    u_same = (flow[..., 0] == flow_u).all()
    v_same = (flow[..., 1] == flow_v).all()
    if number_result:
        compare_u = (flow[..., 0] == flow_u)
        q_u = np.sum(compare_u) / float(compare_u.shape[0] * compare_u.shape[1])
        compare_v = (flow[..., 1] == flow_v)
        q_v = np.sum(compare_v) / float(compare_v.shape[0] * compare_v.shape[1])
        return (q_u + q_v) / 2.0
        # return q_v
    if u_same and v_same:
        return True
    else: 
        return False

def test_list_of(u_folder, v_folder, flow_folder, type_u="jpg", type_v="jpg", type_flow="png", number_result=False):
    list_flow_u = list_image_in_folder(u_folder, type_image=type_u)
    list_flow_v = list_image_in_folder(v_folder, type_image=type_v)
    list_flow = list_image_in_folder(flow_folder, type_image=type_flow)
    result = []
    for u_path, v_path, flow_path in zip(list_flow_u, list_flow_v, list_flow):
        result.append(test_of_image(u_path, v_path, flow_path, number_result=number_result))
    quality = np.sum(result) / float(len(result))
    return quality

# flow_folder = '/media/minhkv/Data/HocTap/Baitap/Python/PythonMachineLearning/data'
# video_name = 'v_ApplyEyeMakeup_g01_c01'
# output_folder = '/media/minhkv/Data/HocTap/Baitap/Python/PythonMachineLearning/data/flow'
# u_folder = os.path.join(flow_folder, 'u', video_name)
# v_folder = os.path.join(flow_folder, 'v', video_name)
# result_test = test_list_of(u_folder, v_folder, output_folder)
# print(test_list_of(u_folder, v_folder, output_folder))