function path = get_lstm_cache_path( trainOrTest, stage, fold, scene, class )

path = '';

if nargin < 1
    fprintf( 'First argument must be one of ''train'', ''test''\n' );
    return;
end
if nargin < 2
    fprintf( 'Second argument must be one of ''labels'', ''features'', ''afe'', ''earsignals''\n' );
    return;
end
if nargin < 3
    fprintf( 'Third argument must be fold id\n' );
    return;
end
if nargin < 4
    fprintf( 'Fourth argument must be scene id\n' );
    return;
end
if nargin < 5
    fprintf( 'Fifth argument unset; using 1 for class id\n' );
end

if strcmpi( trainOrTest, 'train' )
    pathes_ = load( 'mc4_lstm_train_pathes.mat' );
elseif strcmpi( trainOrTest, 'test' )
    pathes_ = load( 'mc4_lstm_test_pathes.mat' );
    fold = fold - 6;
else
    fprintf( 'First argument must be one of ''train'', ''test''\n' );
    return;
end
    
if strcmpi( stage, 'labels' )
    pathes = pathes_.labelCacheDirs;
    idxs = {class, fold, scene};
elseif strcmpi( stage, 'features' )
    pathes = pathes_.featureCacheDirs;
    idxs = {fold, scene};
elseif strcmpi( stage, 'afe' )
    pathes = pathes_.afeCacheDirs;
    idxs = {fold, scene};
elseif strcmpi( stage, 'earsignals' )
    pathes = pathes_.earSignalCacheDirs;
    idxs = {fold, scene};
else
    fprintf( 'Second argument must be one of ''labels'', ''features'', ''afe'', ''earsignals''\n' );
    return;
end

path = pathes{idxs{:}};


end
