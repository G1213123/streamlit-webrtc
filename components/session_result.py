import pandas as pd


def result_to_df(cache, counter_column = False):
    result_df = cache
    result_df = [item for sublist in result_df for item in sublist]
    fields = ['frame_num', 'track_id', 'class_name', 'tlwh', 'confidence']
    result_df = pd.DataFrame(
        [{fn: getattr( f, fn ) for fn in fields} for f in result_df] )
    if 'tlwh' in result_df.columns:
        result_df['top'], result_df['left'], result_df['width'], result_df['height'] = zip(
            *result_df.pop( 'tlwh' ) )
    if counter_column:
        counter_id = [[i] * len( r ) for i, r in enumerate( cache )]
        counter_id = [item for sublist in counter_id for item in sublist]
        result_df.insert( 0, 'counter', counter_id )
    return result_df
