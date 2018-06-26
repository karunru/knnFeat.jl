using MLDataUtils

distance(a::Array{Float64,1}, b::Array{Float64,2}) = sqrt.(vec(sum((a' .- b).^2, 2)))

function get_feat(data::Array{Float64, 2}, X_train::Array{Float64, 2}, y_train, k_index::Int64, class)
    dists = Float64[]
    inclass_X = X_train[y_train .== class, :]

    for i in 1:size(data, 1)
        datum = data[i, :]
        distances = distance(datum, inclass_X)
        sorted_distances_index = sortperm(distances)
        nearest_index = sorted_distances_index[1:k_index]
        dist = sum(distances[nearest_index])
        append!(dists, dist)
    end

    return dists
end

function knnExtract(X, y, k::Int64 = 1, holds::Int64 = 5)
    classes = sort(unique(y))
    CLASS_NUM = length(classes)
    result = zeros(size(X, 1), CLASS_NUM * k)
    train_index, test_index = kfolds(size(X, 1), holds)

    for hold in 1:holds
        X_train, X_test = X[train_index[hold], :], X[test_index[hold], :]
        y_train, y_test = y[train_index[hold]], y[test_index[hold]]

        features = zeros(0, size(X_test, 1))

        for class_index in 1:CLASS_NUM
            for k_index in 1:k
                feat =  get_feat(X_test, X_train, y_train, k_index, classes[class_index])
                features = vcat(features, feat')
            end
        end

        result[test_index[hold], :] = features'
    end

    return result
end
