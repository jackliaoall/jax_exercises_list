import jax.numpy as jnp

m = 20

# 生成数据集x，此时的数据集x是一个二维矩阵
x0 = jnp.ones((m, 1))
x1 = jnp.arange(1, m+1).reshape(m, 1)
x = jnp.hstack((x0, x1)) #【20,2】

y = jnp.array([
    3, 4, 5, 5, 2, 4, 7, 8, 11, 8, 12,
    11, 13, 13, 16, 17, 18, 17, 19, 21
]).reshape(m, 1)

alpha = 0.01

#这里的theta是一个[2,1]大小的矩阵，用来与输入x进行计算获得计算的预测值y_pred,而y_pred是与y计算误差

def error_function(theta,x,y):
    h_pred = jnp.dot(x,theta)
    j_theta = (1./2*m) * jnp.dot(jnp.transpose(h_pred), h_pred)
    return j_theta

def gradient_function(theta, X, y):
    h_pred = jnp.dot(X, theta) - y
    return (1./m) * jnp.dot(jnp.transpose(X), h_pred)


def gradient_descent(X, y, alpha):
    theta = jnp.array([1, 1]).reshape(2, 1)  #[2,1]  这里的theta是参数
    gradient = gradient_function(theta,X,y)
    while not jnp.all(jnp.absolute(gradient) <= 1e-5):
        theta = theta - alpha * gradient
        gradient = gradient_function(theta, X, y)
    return theta

theta = gradient_descent(x, y, alpha)
print('optimal:', theta)
print('error function:', error_function(theta, x, y)[0,0])