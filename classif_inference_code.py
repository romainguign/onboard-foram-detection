def wrap_frozen_graph_tf2(graph_def, inputs, outputs, print_graph=False):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph
    if print_graph == True:
        # print("-" * 50)
        # print("Frozen model layers: ")
        layers = [op.name for op in import_graph.get_operations()]
        # for layer in layers:
        #     print(layer)
        # print("-" * 50)
    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))

def load_frozen_model_tf2(filepath, inputs, outputs):
    with tf.io.gfile.GFile(filepath, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        loaded = graph_def.ParseFromString(f.read())
    # Wrap frozen graph to ConcreteFunctions
    frozen_func = wrap_frozen_graph_tf2(graph_def=graph_def,
                                        inputs=inputs,
                                        outputs=outputs,
                                        print_graph=False)
    # print("-" * 80)
    # print("Frozen model inputs: ")
    # print(frozen_func.inputs)
    # print("Frozen model outputs: ")
    # print(frozen_func.outputs)
    return frozen_func

def load_from_xml(filename, session=None):
    project = ET.parse(filename).getroot()

    protobuf = project.find('protobuf').text
    # print(f"Loading model from {filename}")
    # print("- protobuf: " + protobuf)

    input = None
    output = None
    img_size = np.zeros(3, dtype=int)
    cls_labels = []

    list_xml = project.find('labels')
    # print("- labels:")
    for i, entry_xml in enumerate(list_xml.iter('label')):
        code = entry_xml.find('code').text
        cls_labels.append(code)
        # print(f"  - {code}")

    list_xml = project.find('inputs')
    for i, entry_xml in enumerate(list_xml.iter('input')):
        if i == 0:
            input_name = entry_xml.find('operation').text + ":0"
            img_size[0] = int(entry_xml.find('height').text)
            img_size[1] = int(entry_xml.find('width').text)
            img_size[2] = int(entry_xml.find('channels').text)

    list_xml = project.find('outputs')
    for i, entry_xml in enumerate(list_xml.iter('output')):
        if i == 0:
            output_name = entry_xml.find('operation').text + ":0"

    # Find the 'cnn' tag within 'params'
    cnn_tag = project.find('.//params/cnn')
    img_type = "rgb"
    if cnn_tag is not None and cnn_tag.text:
        txt = cnn_tag.text
        if "'k'" in txt or "'greyscale'" in txt:
            img_type = "k"

    full_protobuf_path = os.path.join(os.path.dirname(filename), protobuf)

    # print(f"- input: {input_name}")
    # print(f"  - height: {img_size[0]}")
    # print(f"  - width: {img_size[1]}")
    # print(f"  - channels: {img_size[2]}")
    # print(f"- output: {output_name}")

    model = load_frozen_model_tf2(full_protobuf_path, input_name, output_name)
    return model, img_size, img_type, cls_labels

def get_image_paths_and_samples(base_path, sample_name="unknown"):
    base_path = Path(base_path)
    image_paths = []
    samples = []
    for subdir in base_path.iterdir():
        if subdir.is_dir():
            for file in subdir.rglob('*'):
                if file.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif'):
                    image_paths.append(str(file))
                    samples.append(subdir.name)
        else:
            file = subdir
            if file.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif'):
                image_paths.append(str(file))
                samples.append(sample_name)
    
    return image_paths, samples



def classify_folder(model_info_path,
                    images_path,
                    output_path,
                    batch_size,
                    sample_name="unknown",
                    unsure_threshold=0.0):
    model, img_size, img_type, labels = load_from_xml(model_info_path)

    # Create a dataset of image paths
    image_paths, sample_names = get_image_paths_and_samples(images_path, sample_name)

    image_dataset = tf.data.Dataset.from_tensor_slices(image_paths)

    # Image loading functions
    def load_and_preprocess_image(image_path):
        def _load_image(image_path):
            image_path = image_path.numpy().decode('utf-8')
            image = load_image(image_path, img_size, img_type)
            image = tf.convert_to_tensor(image, dtype=tf.float32)
            return image
        image = tf.py_function(_load_image, [image_path], tf.float32)
        image.set_shape(img_size)
        return image

    # Map using the image dataset
    image_dataset = image_dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch the dataset
    image_dataset = image_dataset.batch(batch_size)

    # Run predictions
    predictions = []
    idxs = []
    cls = []
    scores = []
    for batch in tqdm(image_dataset):
        preds = model(batch).numpy()
        batch_idxs = np.argmax(preds, axis=1)
        batch_labels = [labels[idx] for idx in batch_idxs]
        batch_scores = np.max(preds, axis=1)
        predictions.extend(preds)
        idxs.extend(batch_idxs)
        cls.extend(batch_labels)
        scores.extend(batch_scores)

    idxs = [idx if score > unsure_threshold else -1 for idx, score in zip(idxs, scores)]
    cls = [cls if score > unsure_threshold else "unsure" for cls, score in zip(cls, scores)]

    df = pd.DataFrame({
        "filename": image_paths,
        "short_filename": [Path(f).relative_to(images_path) for f in image_paths],
        "sample": sample_names,
        "class_index": idxs,
        "class": cls,
        "score": scores
    })
    df.to_csv(output_path, index=False)


train_number = 9
model = "ResNet50_20260413-103632"

crops_dir = "Classification/Crops_wo_duplicates"

model_info_path = f'Classification/Models/zaper_reprise/{model}/model_tf2/network_info.xml'

os.makedirs(f'Classification/outputs/runs/train{train_number}', exist_ok=True)

for zone in os.listdir(crops_dir):
    images_path = os.path.join(crops_dir, zone)
    output_path = f'Classification/outputs/runs/train{train_number}/{zone}_preds.csv'
    batch_size = 16

    print("Zone: ", zone)

    classify_folder(model_info_path,
                        images_path,
                        output_path,
                        batch_size,
                        sample_name="unknown",
                        unsure_threshold=0.5)