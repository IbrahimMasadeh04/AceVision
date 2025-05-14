def convert_pixel_distance_to_meters(pixel_dist, ref_height_in_meter, ref_height_in_pixel):
    return (pixel_dist * ref_height_in_meter) / ref_height_in_pixel

def convert_meters_to_pixel_distance(meter_dist, ref_height_in_meter, ref_height_in_pixel):
    return (meter_dist * ref_height_in_pixel) / ref_height_in_meter